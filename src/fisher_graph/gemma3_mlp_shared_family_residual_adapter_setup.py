"""Freeze the first Gemma shared-family compression experiment.

This setup is the standard-library-only successor to the authenticated exact
heterogeneous generator-family lift.  It preregisters one deliberately narrow
causal variable: two nonadjacent MLP invocations share one exact anchor core,
while the nonanchor receives either an exact dense residual, no residual, or a
rank-reduced residual adapter.  Every other model coefficient, invocation,
carrier edge occurrence, and terminal operation remains exact.

Building, validating, loading, or writing this setup does not import a tensor
library, load Gemma, open the prompt panel, tokenize text, fit an adapter,
compile a model, or execute an experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_SCHEMA = (
    "fisher_graph.gemma3_mlp_shared_family_residual_adapter.setup.v1"
)
GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-mlp-shared-family-residual-adapter-setup-v1.json"
)

_SETUP_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-residual-adapter-setup:v1\0"
)
_ROLE_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-role-manifest:v1\0"
)
_EXAMPLE_ID_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-example-ids:v1\0"
)

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_SCALARS = 268_098_176
PINNED_MLP_INVOCATION_SCALARS = 3_933_440
PINNED_MLP_MATRIX_SCALARS = 3_932_160
PINNED_MLP_NORM_SCALARS = 1_280
PINNED_MLP_RESIDUAL_RANK_SLOPE = 8_064
PINNED_HIDDEN_WIDTH = 640
PINNED_MLP_WIDTH = 2_048
PINNED_LAYER_COUNT = 18
PINNED_EVENT_COUNT = 39
PINNED_MUTATION_COUNT = 36
PINNED_EDGE_OCCURRENCE_COUNT = 75

PINNED_PREDECESSOR_PROTOCOL_PATH = Path(
    "artifacts/research/"
    "gemma3-heterogeneous-whole-model-generator-family-protocol-v1.json"
)
PINNED_PREDECESSOR_PROTOCOL_SHA256 = (
    "32747a59ce2aa7ad0633904b54680d19b7f2da1545bb136c4816d025667f61f2"
)
PINNED_PREDECESSOR_PROTOCOL_FILE_SHA256 = (
    "b2caaae36742864f7a0bb491ed80ab08f8c94029d6d554d7d68cdbd1f7b1d243"
)
PINNED_PREDECESSOR_REPORT_PATH = Path(
    ".local-runs/google--gemma-3-270m/"
    "heterogeneous-generator-family-exact-v1.json"
)
PINNED_PREDECESSOR_REPORT_SHA256 = (
    "7f062b40a2af2fe61364b208189773b82569bbe88902c0b8f3e7bbe607401926"
)
PINNED_PREDECESSOR_REPORT_FILE_SHA256 = (
    "2ad135209ff1c780944c7d0efd5dbd7b52724266dc880345abb31cfc9bbcf64a"
)

PINNED_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_PANEL_ID = "gemma3-single-token-forced-choice-v1"

ORDERED_ROLES = ("A_fit", "A_select", "B_guard", "C_assessment")
ORDERED_RESIDUAL_RANKS: tuple[str | int, ...] = (
    "dense",
    256,
    128,
    64,
    32,
)
LOW_RANK_POINTS = (256, 128, 64, 32)

SEEDS = {
    "base": 1_729,
    "pair_search": 1_730,
    "adapter_fit": 1_731,
    "matched_control_fit": 1_732,
    "bootstrap": 1_733,
    "deterministic_replay": 1_734,
}

_FAMILY_PREFIXES = {
    "integer_addition": "addition",
    "integer_subtraction": "subtraction",
    "integer_comparison": "comparison",
    "lexical_antonym": "antonym",
    "category_completion": "category",
    "world_capitals": "capital",
}
_ROLE_FAMILIES = {
    "A_fit": ("integer_addition", "integer_subtraction"),
    "A_select": ("integer_comparison",),
    "B_guard": ("lexical_antonym", "category_completion"),
    "C_assessment": ("world_capitals",),
}
_ROLE_PERMISSIONS = {
    "A_fit": {
        "may_select_pair_and_anchor": True,
        "may_fit_coefficients": True,
        "may_select_rank_for_promotion": False,
        "may_open_later_roles": False,
    },
    "A_select": {
        "may_select_pair_and_anchor": False,
        "may_fit_coefficients": False,
        "may_select_rank_for_promotion": False,
        "may_freeze_pareto_frontier": True,
    },
    "B_guard": {
        "may_select_pair_and_anchor": False,
        "may_fit_coefficients": False,
        "may_select_rank_for_promotion": True,
        "opened_once": True,
    },
    "C_assessment": {
        "may_select_pair_and_anchor": False,
        "may_fit_coefficients": False,
        "may_select_rank_for_promotion": False,
        "opened_once_after_guard_pass": True,
    },
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _setup_sha256(value: Mapping[str, object]) -> str:
    return _sha256(_SETUP_HASH_DOMAIN, value)


def _family_example_ids(family_id: str) -> list[str]:
    prefix = _FAMILY_PREFIXES[family_id]
    return [f"{prefix}-{ordinal:02d}" for ordinal in range(1, 11)]


def _role_manifests() -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for role_id in ORDERED_ROLES:
        family_ids = _ROLE_FAMILIES[role_id]
        example_ids = [
            example_id
            for family_id in family_ids
            for example_id in _family_example_ids(family_id)
        ]
        manifest: dict[str, object] = {
            "role_id": role_id,
            "panel_id": PINNED_PANEL_ID,
            "panel_path": str(PINNED_PANEL_PATH),
            "panel_sha256": PINNED_PANEL_SHA256,
            "family_ids": list(family_ids),
            "family_count": len(family_ids),
            "ordered_example_ids": example_ids,
            "example_count": len(example_ids),
            "example_ids_sha256": _sha256(
                _EXAMPLE_ID_HASH_DOMAIN,
                example_ids,
            ),
            "permissions": _ROLE_PERMISSIONS[role_id],
        }
        manifest["role_manifest_sha256"] = _sha256(
            _ROLE_HASH_DOMAIN,
            manifest,
        )
        manifests[role_id] = manifest
    return manifests


def mlp_residual_adapter_scalars(residual_rank: int) -> int:
    if type(residual_rank) is not int or residual_rank <= 0:
        raise ValueError("residual rank must be a positive integer")
    return (
        PINNED_MLP_NORM_SCALARS
        + PINNED_MLP_RESIDUAL_RANK_SLOPE * residual_rank
    )


def _resource_point(residual_rank: str | int) -> dict[str, object]:
    fixed_exact_outside_pair = (
        PINNED_SOURCE_SCALARS - 2 * PINNED_MLP_INVOCATION_SCALARS
    )
    if residual_rank == "dense":
        adapter_scalars = PINNED_MLP_INVOCATION_SCALARS
        representation = "exact_dense_parameter_residual"
        exact_control = True
    elif type(residual_rank) is int and residual_rank in LOW_RANK_POINTS:
        adapter_scalars = mlp_residual_adapter_scalars(residual_rank)
        representation = f"three_low_rank_matrix_residuals_r{residual_rank}"
        exact_control = False
    else:
        raise ValueError("residual rank is not preregistered")
    candidate = (
        fixed_exact_outside_pair
        + PINNED_MLP_INVOCATION_SCALARS
        + adapter_scalars
    )
    removed = PINNED_SOURCE_SCALARS - candidate
    return {
        "point_id": f"shared_core.residual_{residual_rank}",
        "residual_rank": residual_rank,
        "representation": representation,
        "exact_dense_control": exact_control,
        "fixed_exact_outside_pair_scalars": fixed_exact_outside_pair,
        "shared_anchor_core_scalars": PINNED_MLP_INVOCATION_SCALARS,
        "nonanchor_residual_adapter_scalars": adapter_scalars,
        "candidate_stored_scalars": candidate,
        "removed_scalars": removed,
        "whole_model_reduction_fraction": removed / PINNED_SOURCE_SCALARS,
        "physical_shared_core_count": 1,
        "physical_nonanchor_dense_MLP_count": 0,
        "nonanchor_residual_adapter_count": 1,
        "learned_correction_edge_count": 0,
    }


def _zero_residual_point() -> dict[str, object]:
    candidate = PINNED_SOURCE_SCALARS - PINNED_MLP_INVOCATION_SCALARS
    removed = PINNED_SOURCE_SCALARS - candidate
    return {
        "point_id": "ablation.shared_core.zero_residual",
        "residual_rank": 0,
        "representation": "anchor_core_invoked_at_both_selected_occurrences",
        "control_only": True,
        "promotable": False,
        "shared_anchor_core_scalars": PINNED_MLP_INVOCATION_SCALARS,
        "nonanchor_residual_adapter_scalars": 0,
        "candidate_stored_scalars": candidate,
        "removed_scalars": removed,
        "whole_model_reduction_fraction": removed / PINNED_SOURCE_SCALARS,
        "learned_correction_edge_count": 0,
    }


def _arm_catalog() -> list[dict[str, object]]:
    arms: list[dict[str, object]] = [
        {
            "arm_id": "native",
            "arm_kind": "exact_oracle",
            "fit_allowed": False,
            "promotable": False,
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        {
            "arm_id": "exact.heterogeneous_full_rank_atom_bank_lift",
            "arm_kind": "authenticated_predecessor_exact_control",
            "fit_allowed": False,
            "promotable": False,
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        {
            "arm_id": "exact.shared_core_dense_residual",
            "arm_kind": "successor_literal_parity_control",
            "fit_allowed": False,
            "promotable": False,
            **_resource_point("dense"),
        },
        {
            "arm_id": "ablation.shared_core_zero_residual",
            "arm_kind": "shared_core_without_specialization_control",
            "fit_allowed": False,
            **_zero_residual_point(),
        },
    ]
    for rank_index, rank in enumerate(LOW_RANK_POINTS):
        point = _resource_point(rank)
        arms.extend(
            (
                {
                    "arm_id": f"primary.shared_core_residual_r{rank}",
                    "arm_kind": "primary_shared_family_candidate",
                    "fit_allowed": True,
                    "promotable": True,
                    "fit_nonce": f"primary.shared_core_residual_r{rank}.v1",
                    "fit_seed": SEEDS["adapter_fit"] + 100 * rank_index,
                    **point,
                },
                {
                    "arm_id": f"control.independent_member_r{rank}",
                    "arm_kind": "byte_matched_independent_factorization",
                    "fit_allowed": True,
                    "promotable": False,
                    "control_only": True,
                    "fit_nonce": f"control.independent_member_r{rank}.v1",
                    "fit_seed": (
                        SEEDS["matched_control_fit"] + 100 * rank_index
                    ),
                    "shared_core_used_by_nonanchor": False,
                    "nonanchor_representation": (
                        "standalone_three_matrix_low_rank_MLP_with_exact_norms"
                    ),
                    **point,
                },
                {
                    "arm_id": f"control.shuffled_residual_r{rank}",
                    "arm_kind": "invocation_assignment_shuffle",
                    "fit_allowed": False,
                    "promotable": False,
                    "control_only": True,
                    "shuffle_rule": (
                        "attach_the_frozen_nonanchor_adapter_to_the_anchor_"
                        "occurrence_and_leave_the_nonanchor_with_zero_residual"
                    ),
                    **point,
                },
            )
        )
    return arms


def _build_without_hash() -> dict[str, object]:
    roles = _role_manifests()
    arms = _arm_catalog()
    exact_candidate_values = [
        _resource_point(rank)["candidate_stored_scalars"]
        for rank in ORDERED_RESIDUAL_RANKS
    ]
    return {
        "schema": GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_SCHEMA,
        "format_version": (
            GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_FORMAT_VERSION
        ),
        "artifact_role": (
            "setup_only_first_nonadjacent_MLP_shared_core_residual_adapter_"
            "compression_experiment"
        ),
        "setup_receipt": {
            "predecessor_artifacts_opened_by_builder": False,
            "model_weights_loaded": False,
            "prompt_panel_opened": False,
            "prompt_text_opened": False,
            "tokenization_performed": False,
            "activation_or_Fisher_capture_performed": False,
            "pair_or_anchor_selected": False,
            "adapter_or_control_fitted": False,
            "compiler_executed": False,
            "model_forward_executed": False,
            "experiment_report_written": False,
            "new_experiment_metrics_present": False,
        },
        "predecessor_chain": {
            "protocol": {
                "path": str(PINNED_PREDECESSOR_PROTOCOL_PATH),
                "schema": (
                    "fisher_graph.gemma3_heterogeneous_whole_model_generator_"
                    "family.protocol.v1"
                ),
                "protocol_sha256": PINNED_PREDECESSOR_PROTOCOL_SHA256,
                "file_sha256": PINNED_PREDECESSOR_PROTOCOL_FILE_SHA256,
            },
            "report": {
                "path": str(PINNED_PREDECESSOR_REPORT_PATH),
                "schema": (
                    "fisher_graph.gemma3_heterogeneous_generator_family."
                    "experiment.v1"
                ),
                "report_sha256": PINNED_PREDECESSOR_REPORT_SHA256,
                "file_sha256": PINNED_PREDECESSOR_REPORT_FILE_SHA256,
            },
            "required_predecessor_arm_id": (
                "exact.heterogeneous_full_rank_atom_bank_lift"
            ),
            "required_predecessor_outcome": {
                "all_five_arms_passed": True,
                "maximum_valid_query_logit_error": 0.0,
                "delta_nll_per_token": 0.0,
                "native_to_candidate_kl_per_query": 0.0,
                "top1_agreement": 1.0,
                "compiled_source_calls": 0,
                "invocation_count": PINNED_EVENT_COUNT,
                "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
                "stored_scalars": PINNED_SOURCE_SCALARS,
            },
            "both_internal_and_file_SHA256_values_verified_before_model_load": (
                True
            ),
            "predecessor_metrics_may_select_pair_rank_or_threshold": False,
        },
        "model_contract": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "MLP_width": PINNED_MLP_WIDTH,
            "layer_count": PINNED_LAYER_COUNT,
            "event_count": PINNED_EVENT_COUNT,
            "mutation_count": PINNED_MUTATION_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "source_unique_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        "scope_freeze": {
            "compressed_component": "one_nonadjacent_MLP_pair_only",
            "tied_lexical_ingress_and_egress": "exact_and_frozen",
            "all_18_attention_invocations": "exact_and_frozen",
            "unselected_16_MLP_invocations": "exact_singletons_and_frozen",
            "selected_anchor_MLP": "one_exact_shared_core",
            "selected_nonanchor_MLP": (
                "shared_core_plus_one_invocation_specific_residual_adapter"
            ),
            "identity_carrier_edges": "all_36_unchanged_and_parameter_free",
            "primary_generator_occurrences": "all_36_ordered_and_retained",
            "terminal_RMSNorm_and_logit_softcap": "exact_and_frozen",
            "invocation_count": PINNED_EVENT_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "learned_correction_edge_count": 0,
            "optional_correction_edge_budget_present": False,
            "charts_or_learned_transports_present": False,
            "changing_edges_topology_or_other_families_requires_new_protocol": (
                True
            ),
        },
        "data_firewall": {
            "panel_path": str(PINNED_PANEL_PATH),
            "panel_sha256": PINNED_PANEL_SHA256,
            "panel_family_count": 6,
            "panel_example_count": 60,
            "ordered_roles": list(ORDERED_ROLES),
            "role_allocation_rule": (
                "whole_families_in_canonical_panel_order_2_1_2_1;"
                "no_metric_or_model_observation_used"
            ),
            "roles": roles,
            "prompt_example_and_family_overlap_across_roles_allowed": False,
            "all_60_panel_examples_assigned_exactly_once": True,
            "role_manifest_SHA256_values_frozen_by_this_setup": True,
            "all_role_manifests_verified_before_model_load": True,
            "raw_prompt_text_or_token_ids_written_to_setup": False,
            "A_fit_is_the_only_role_visible_to_pair_search_or_fitting": True,
            "A_select_may_refit_or_change_pair_anchor_or_topology": False,
            "pareto_frontier_frozen_and_hashed_before_B_guard": True,
            "B_guard_opened_once_and_scores_entire_frontier": True,
            "B_guard_may_refit_reorder_prune_or_change_thresholds": False,
            "C_assessment_opened_once_after_one_guard_winner": True,
            "C_assessment_may_refit_select_or_change_thresholds": False,
            "failed_guard_or_assessment_requires_new_protocol_and_splits": True,
        },
        "pair_and_anchor_selection": {
            "role": "A_fit",
            "eligible_family_id": "family.mlp.gated_gelu_h640_i2048",
            "eligible_block_ordinals": list(range(PINNED_LAYER_COUNT)),
            "eligible_invocation_ids": [
                f"mutation.{2 * block + 2}"
                for block in range(PINNED_LAYER_COUNT)
            ],
            "nonadjacent_definition": "absolute_block_difference_at_least_2",
            "unordered_eligible_pair_count": 136,
            "both_anchor_orientations_evaluated": True,
            "directed_pair_and_anchor_candidate_count": 272,
            "capture_trajectory": (
                "exact_predecessor_A_fit_input_at_the_nonanchor_invocation"
            ),
            "row_weight": (
                "squared_nonanchor_MLP_output_score_gradient_energy_"
                "normalized_to_mean_one_within_each_A_fit_family"
            ),
            "family_aggregation": "equal_mean_across_A_fit_families",
            "primary_objective_name": (
                "zero_residual_shared_core_Fisher_weighted_relative_SSE"
            ),
            "primary_objective_direction": "minimize",
            "primary_objective_formula": (
                "equal_family_mean(sum_rows(w*||exact_member_delta-"
                "anchor_core_delta||^2)/max(sum_rows(w*||exact_member_delta||^2),"
                "epsilon))"
            ),
            "comparison_rule": (
                "evaluate_exact_member_and_exact_anchor_core_on_the_same_"
                "nonanchor_exact_A_fit_input_rows_without_fitting_an_adapter"
            ),
            "epsilon": "2^-52_times_max(1,target_Fisher_weighted_energy)",
            "ordered_tie_break": [
                "lexicographically_smaller_SHA256_of_canonical_float32_"
                "aligned_design_and_target_tensor_bytes",
                "fail_closed_if_the_numerical_tensor_digest_is_identical",
            ],
            "adapter_probe_fit_during_pair_selection_allowed": False,
            "activation_and_Fisher_evidence_only": True,
            "block_ordinal_or_distance_used_as_numerical_feature": False,
            "block_or_invocation_label_may_break_a_numerical_tie": False,
            "nonadjacency_is_eligibility_only": True,
            "pair_anchor_and_A_fit_row_manifest_frozen_before_A_select": True,
            "pair_assignment_manifest_SHA256_required": True,
        },
        "fit_contract": {
            "one_separately_materialized_artifact_per_arm": True,
            "supernet_or_nested_factor_slicing_allowed": False,
            "source_model_parameters_gradients_and_training_mode_immutable": True,
            "selected_member_prefix_is_exact_by_single_approximation": True,
            "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
            "candidate_prefix_recapture_required": False,
            "shared_core": "exact_copied_anchor_MLP_including_two_RMSNorms",
            "dense_residual_control": (
                "source_sized_exact_nonanchor_owned_adapter_with_the_shared_"
                "core_plus_dense_residual_relationship_audited_observationally"
            ),
            "dense_residual_control_is_not_low_rank_r640": True,
            "dense_residual_control_live_core_plus_residual_arithmetic": False,
            "dense_residual_control_reason": (
                "delegate_one_exact_nonanchor_operator_call_to_preserve_the_"
                "literal_parity_gate_without_a_second_rounding_path"
            ),
            "low_rank_residual": {
                "matrix_names": ["gate_proj", "up_proj", "down_proj"],
                "factor_shape_per_matrix": ["output_by_r", "r_by_input"],
                "per_matrix_factor_scalars": "2688*r",
                "all_three_matrix_factor_scalars": "8064*r",
                "two_RMSNorm_residual_vectors_kept_dense": True,
                "RMSNorm_residual_scalars": PINNED_MLP_NORM_SCALARS,
                "adapter_scalar_formula": "1280+8064*r",
                "fit_objective": (
                    "per_projection_A_fit_Fisher_weighted_reduced_rank_"
                    "regression"
                ),
                "gate_and_up_design": (
                    "X_equals_exact_member_normalized_MLP_input"
                ),
                "down_design": "H_equals_exact_member_gated_activation",
                "primary_target": (
                    "design_times_(W_member-W_anchor)_transpose"
                ),
                "matched_independent_target": (
                    "design_times_W_member_transpose"
                ),
                "row_weight": (
                    "normalized_squared_projection_output_gradient_energy"
                ),
                "solver_order": [
                    "multiply_design_and_target_rows_by_sqrt_weight",
                    "form_ridge_gram_and_solve_the_full_weighted_coefficient",
                    "left_whiten_the_coefficient_by_the_ridge_gram_Cholesky",
                    "compute_canonical_rank_r_SVD_of_the_whitened_coefficient",
                    "unwhiten_the_truncated_coefficient",
                    "factor_the_rank_r_coefficient_as_output_by_r_times_r_by_input",
                ],
                "relative_ridge": 1.0e-4,
                "ridge_scale": (
                    "trace(X_weighted_transpose_X_weighted)_divided_by_"
                    "design_width"
                ),
                "absolute_ridge": "relative_ridge_times_ridge_scale",
                "solver_dtype": "float64",
                "stored_factor_dtype": "source_parameter_dtype",
                "factor_gauge": (
                    "first_maximum_magnitude_entry_of_each_SVD_output_vector_"
                    "is_positive_and_the_unwhitened_input_factor_uses_the_"
                    "same_sign"
                ),
                "fit_rank_order": list(LOW_RANK_POINTS),
                "factors_for_each_rank_fitted_independently": True,
                "fit_nonce_and_seed_unique_per_rank_and_arm": True,
                "optimizer_or_gradient_descent_used": False,
            },
            "zero_residual_control": (
                "invoke_the_exact_anchor_core_at_both_selected_occurrences"
            ),
            "byte_matched_independent_control": (
                "keep_anchor_exact_and_fit_nonanchor_as_a_standalone_three_"
                "matrix_low_rank_MLP_with_exact_norms_at_the_same_rank_and_"
                "scalar_budget"
            ),
            "shuffled_control": (
                "move_the_frozen_nonanchor_adapter_to_the_anchor_occurrence_"
                "and_leave_the_nonanchor_occurrence_without_an_adapter"
            ),
            "no_control_may_change_rows_rank_budget_or_optimizer": True,
            "pair_search_fits_no_adapter": True,
            "rank_ladder_fit_begins_only_after_pair_anchor_manifest_freezes": (
                True
            ),
            "seeds": SEEDS,
        },
        "arm_contract": {
            "residual_rank_order": list(ORDERED_RESIDUAL_RANKS),
            "arm_order": [arm["arm_id"] for arm in arms],
            "arms": arms,
            "execution_gate_chain": [
                "native",
                "exact.heterogeneous_full_rank_atom_bank_lift",
                "exact.shared_core_dense_residual",
                "compressed_and_control_arms_in_frozen_order",
            ],
            "dense_exact_control_must_pass_before_any_compressed_fit": True,
            "zero_residual_arm_is_control_only_despite_highest_reduction": True,
            "primary_rank_selection_direction": (
                "highest_compression_passing_rank_32_then_64_then_128_then_256"
            ),
        },
        "resource_accounting": {
            "source_unique_stored_scalars": PINNED_SOURCE_SCALARS,
            "source_MLP_scalars_per_invocation": (
                PINNED_MLP_INVOCATION_SCALARS
            ),
            "source_MLP_matrix_scalars_per_invocation": PINNED_MLP_MATRIX_SCALARS,
            "source_MLP_norm_scalars_per_invocation": PINNED_MLP_NORM_SCALARS,
            "fixed_exact_outside_selected_pair_scalars": (
                PINNED_SOURCE_SCALARS - 2 * PINNED_MLP_INVOCATION_SCALARS
            ),
            "one_shared_core_scalars": PINNED_MLP_INVOCATION_SCALARS,
            "dense_residual_scalars": PINNED_MLP_INVOCATION_SCALARS,
            "low_rank_residual_formula": "1280+8064*r",
            "whole_model_candidate_formula": (
                "source-2*3933440+3933440+residual_adapter_scalars"
            ),
            "ordered_candidate_stored_scalars": exact_candidate_values,
            "zero_residual_candidate_stored_scalars": (
                PINNED_SOURCE_SCALARS - PINNED_MLP_INVOCATION_SCALARS
            ),
            "ordered_points": [
                _resource_point(rank) for rank in ORDERED_RESIDUAL_RANKS
            ],
            "every_matched_control_has_identical_rank_point_scalar_budget": True,
            "parameters_and_persistent_buffers_counted": True,
            "equal_values_without_physical_alias_are_deduplicated": False,
            "nonanchor_dense_MLP_must_be_physically_absent_in_compressed_arms": (
                True
            ),
            "candidate_source_storage_disjoint_except_internal_tied_aliases": (
                True
            ),
            "learned_correction_edge_scalars": 0,
            "charts_and_learned_transport_scalars": 0,
            "logical_MACs_serialized_bytes_peak_bytes_and_latency_reported_"
            "separately": True,
            "pair_pilot_is_not_required_to_meet_twenty_percent_global_target": (
                True
            ),
        },
        "acceptance_gates": {
            "exact_dense_residual_control": {
                "maximum_valid_query_logit_absolute_error": 0.0,
                "absolute_delta_nll_per_token_maximum": 0.0,
                "native_to_candidate_kl_per_query_maximum": 0.0,
                "top1_agreement_minimum": 1.0,
                "all_36_mutation_input_and_delta_hashes_exact": True,
                "all_36_carrier_boundary_hashes_exact": True,
                "source_projection_and_fallback_calls": 0,
                "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
            },
            "compressed_candidate_fidelity": {
                "delta_nll_per_token_maximum": 0.10,
                "native_to_candidate_kl_per_query_maximum": 0.10,
                "top1_agreement_minimum": 0.90,
                "valid_query_logit_nrmse_maximum": 0.10,
                "native_correct_downstream_accuracy_retained_minimum": 0.90,
                "worst_family_bootstrap_lower_bound_minimum": 0.85,
                "selected_nonanchor_MLP_delta_nrmse_maximum": 0.25,
                "selected_nonanchor_MLP_delta_cosine_minimum": 0.90,
                "final_carrier_nrmse_maximum": 0.10,
                "all_36_carrier_boundary_metrics_reported": True,
                "first_divergence_occurrence_reported": True,
                "source_projection_and_fallback_calls": 0,
            },
            "control_dominance": {
                "real_residual_gain_definition": (
                    "zero_residual_delta_nll_minus_primary_delta_nll"
                ),
                "real_residual_gain_simultaneous_lower_bound_minimum_exclusive": (
                    0.0
                ),
                "gain_over_matched_independent_definition": (
                    "independent_delta_nll_minus_primary_delta_nll"
                ),
                "gain_over_matched_independent_simultaneous_lower_bound_"
                "minimum_exclusive": 0.0,
                "shuffled_gain_definition": (
                    "zero_residual_delta_nll_minus_shuffled_delta_nll"
                ),
                "shuffled_gain_fraction_of_real_simultaneous_upper_bound_"
                "maximum": 0.25,
                "real_residual_gain_over_zero_and_matched_independent_required": (
                    True
                ),
                "same_direction_reported_for_KL_and_nonanchor_delta_nrmse": True,
                "no_single_metric_tradeoff_may_override_a_failed_hard_gate": True,
            },
            "B_guard_bootstrap": {
                "role": "B_guard",
                "seed": SEEDS["bootstrap"],
                "resamples": 10_000,
                "confidence_level": 0.95,
                "sidedness": "one_sided",
                "cluster_unit": "prompt_or_paraphrase_family",
                "family_weighting": "equal_family_mean",
                "paired_resample_indices_shared_by_all_arms_and_metrics": True,
                "simultaneous_method": (
                    "centered_max_deviation_across_all_ranks_metrics_and_"
                    "control_contrasts"
                ),
                "upper_bounds_used_for_delta_nll_KL_nrmse": True,
                "lower_bounds_used_for_top1_accuracy_cosine_and_control_"
                "margins": True,
                "entire_frozen_frontier_scored_once": True,
                "winner_rule": (
                    "highest_compression_primary_rank_passing_every_"
                    "simultaneous_fidelity_control_runtime_and_resource_gate"
                ),
            },
            "C_assessment": {
                "candidate_count": 1,
                "opened_only_after_B_guard_pass": True,
                "same_point_fidelity_thresholds_as_B_guard": True,
                "no_refit_selection_threshold_or_pair_change": True,
                "assessment_failure_revokes_mechanism_qualification": True,
            },
            "runtime_and_resource": {
                "all_39_invocations_and_75_occurrences_retained": True,
                "learned_correction_edge_count": 0,
                "candidate_stored_scalars_must_equal_frozen_formula": True,
                "candidate_stored_scalars_must_be_below_source_for_primary": (
                    True
                ),
                "nonanchor_dense_source_storage_absent": True,
                "no_live_source_or_fallback_calls": True,
                "source_and_candidate_fingerprints_unchanged": True,
                "deterministic_replay_bitwise_identical": True,
            },
        },
        "required_report": {
            "role_manifest_and_row_identity_SHA256_values": True,
            "selected_pair_anchor_and_numerical_tie_break_receipt": True,
            "all_arm_artifact_state_SHA256_values": True,
            "per_role_native_and_arm_NLL_KL_top1_logit_metrics": True,
            "per_event_selected_pair_input_delta_and_carrier_metrics": True,
            "all_36_carrier_boundary_metrics": True,
            "bootstrap_draw_catalog_SHA256": True,
            "unique_scalar_inventory_and_physical_alias_receipt": True,
            "logical_MAC_serialized_byte_peak_memory_and_latency_fields": True,
            "source_projection_fallback_and_edge_call_audit": True,
            "raw_prompt_text_or_token_ids_present": False,
        },
        "claim_boundary": {
            "setup_only": True,
            "model_or_prompt_access_occurred": False,
            "fitting_or_evaluation_occurred": False,
            "compression_success_claimed": False,
            "pair_pilot_can_qualify_shared_family_mechanism_only": True,
            "pair_pilot_cannot_claim_twenty_percent_whole_model_compression": (
                True
            ),
            "parameter_reduction_does_not_imply_byte_MAC_latency_or_memory_"
            "reduction": True,
            "learned_edges_attention_lexical_or_family_growth_require_new_"
            "protocol": True,
        },
    }


def _validate_built_setup(value: Mapping[str, object]) -> None:
    roles = value["data_firewall"]["roles"]
    if list(roles) != list(ORDERED_ROLES):
        raise ValueError("role manifest order drifted")
    all_families: list[str] = []
    all_examples: list[str] = []
    for role_id in ORDERED_ROLES:
        manifest = roles[role_id]
        all_families.extend(manifest["family_ids"])
        all_examples.extend(manifest["ordered_example_ids"])
    if len(all_families) != 6 or len(set(all_families)) != 6:
        raise ValueError("role family manifests are not disjoint and exhaustive")
    if len(all_examples) != 60 or len(set(all_examples)) != 60:
        raise ValueError("role example manifests are not disjoint and exhaustive")
    if [roles[role]["example_count"] for role in ORDERED_ROLES] != [20, 10, 20, 10]:
        raise ValueError("role example counts drifted")

    resources = value["resource_accounting"]
    if resources["ordered_candidate_stored_scalars"] != [
        268_098_176,
        266_230_400,
        265_198_208,
        264_682_112,
        264_424_064,
    ]:
        raise ValueError("shared-family resource ladder drifted")
    if resources["zero_residual_candidate_stored_scalars"] != 264_164_736:
        raise ValueError("zero-residual resource control drifted")
    if value["scope_freeze"]["learned_correction_edge_count"] != 0:
        raise ValueError("learned correction edges are forbidden")

    arm_order = value["arm_contract"]["arm_order"]
    expected = [
        "native",
        "exact.heterogeneous_full_rank_atom_bank_lift",
        "exact.shared_core_dense_residual",
        "ablation.shared_core_zero_residual",
    ]
    for rank in LOW_RANK_POINTS:
        expected.extend(
            (
                f"primary.shared_core_residual_r{rank}",
                f"control.independent_member_r{rank}",
                f"control.shuffled_residual_r{rank}",
            )
        )
    if arm_order != expected:
        raise ValueError("experiment arm order drifted")


def build_gemma3_mlp_shared_family_residual_adapter_setup() -> dict[str, object]:
    setup = _build_without_hash()
    _validate_built_setup(setup)
    setup["setup_sha256"] = _setup_sha256(setup)
    return setup


def validate_gemma3_mlp_shared_family_residual_adapter_setup(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("MLP shared-family setup must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if materialized != build_gemma3_mlp_shared_family_residual_adapter_setup():
        raise ValueError("MLP shared-family setup is not canonical")


def load_gemma3_mlp_shared_family_residual_adapter_setup(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_mlp_shared_family_residual_adapter_setup(value)
    return value


def write_gemma3_mlp_shared_family_residual_adapter_setup(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(setup, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite an MLP shared-family residual-adapter setup"
        ) from error
    return setup


@dataclass(frozen=True, slots=True)
class Gemma3MLPSharedFamilyResidualAdapterSetup:
    """Typed wrapper around the immutable successor setup artifact."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_mlp_shared_family_residual_adapter_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3MLPSharedFamilyResidualAdapterSetup":
        return cls(build_gemma3_mlp_shared_family_residual_adapter_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.state, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing canonical setup without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        load_gemma3_mlp_shared_family_residual_adapter_setup(args.output)
    else:
        write_gemma3_mlp_shared_family_residual_adapter_setup(args.output)
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_FORMAT_VERSION",
    "GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_SETUP_SCHEMA",
    "Gemma3MLPSharedFamilyResidualAdapterSetup",
    "LOW_RANK_POINTS",
    "ORDERED_RESIDUAL_RANKS",
    "ORDERED_ROLES",
    "PINNED_PREDECESSOR_PROTOCOL_SHA256",
    "PINNED_PREDECESSOR_REPORT_SHA256",
    "SEEDS",
    "build_gemma3_mlp_shared_family_residual_adapter_setup",
    "load_gemma3_mlp_shared_family_residual_adapter_setup",
    "main",
    "mlp_residual_adapter_scalars",
    "validate_gemma3_mlp_shared_family_residual_adapter_setup",
    "write_gemma3_mlp_shared_family_residual_adapter_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
