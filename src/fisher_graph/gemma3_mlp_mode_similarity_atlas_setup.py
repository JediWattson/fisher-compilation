"""Freeze the no-compression Gemma MLP-first mode-similarity atlas.

This standard-library-only setup treats source MLP, attention, and lexical
labels as provenance for typed execution modes.  The present atlas admits only
the eighteen token-local carrier-mutation modes whose full native operators
share one cross-execution ABI.  It measures their exact descriptive geometry;
it does not merge, compress, fit, replace, or reconnect anything.

Building, validating, loading, or writing this setup does not import a tensor
library, load Gemma, open prompt text, tokenize, capture a carrier, compute a
gradient, execute a source atom, form a distance matrix, or seriate modes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_SCHEMA = (
    "fisher_graph.gemma3_mlp_mode_similarity_atlas.setup.v1"
)
GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-mlp-mode-similarity-atlas-setup-v1.json"
)

_SETUP_HASH_DOMAIN = b"fisher_graph:gemma3-mlp-mode-similarity-atlas-setup:v1\0"
_FAMILY_MANIFEST_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mode-atlas-family-manifest:v1\0"
)
_PROBE_DRAW_HASH_DOMAIN = b"fisher_graph:gemma3-mode-atlas-probe-draw:v1\0"
_BOOTSTRAP_DRAW_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mode-atlas-bootstrap-draw:v1\0"
)
_BOOTSTRAP_CATALOG_HASH_DOMAIN = (
    b"fisher_graph.mode_similarity_atlas.bootstrap_catalog.v1\0"
)

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_MODEL_SCALARS = 268_098_176
PINNED_HIDDEN_WIDTH = 640
PINNED_LAYER_COUNT = 18
PINNED_MUTATION_COUNT = 36
PINNED_EVENT_COUNT = 39
PINNED_EDGE_OCCURRENCE_COUNT = 75

PINNED_PREDECESSOR_SETUP_PATH = Path(
    "artifacts/research/"
    "gemma3-mlp-shared-family-residual-adapter-setup-v1.json"
)
PINNED_PREDECESSOR_SETUP_SHA256 = (
    "39d4debc38f680d8318bced5b9f553fe9ec8aa9866a990d9852c10e8f55059e8"
)
PINNED_PREDECESSOR_SETUP_FILE_SHA256 = (
    "6fe26fe3e6d4af2460dc2c65e636559b6d00a9b0718c48edd947996692d1f56f"
)
PINNED_PREDECESSOR_REPORT_PATH = Path(
    ".local-runs/google--gemma-3-270m/"
    "mlp-shared-family-residual-adapter-v1.json"
)
PINNED_PREDECESSOR_REPORT_SHA256 = (
    "11bec66d79aaf585c6dbfe835485c3b2aba15a2be5dcd10189bc9d26455f68b4"
)
PINNED_PREDECESSOR_REPORT_FILE_SHA256 = (
    "b29e5e6456793776809acca5d2334700ef16f85b62af257ac51ce936e5fee43d"
)

PINNED_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_PANEL_ID = "gemma3-single-token-forced-choice-v1"

ORDERED_FAMILY_IDS = (
    "integer_addition",
    "integer_subtraction",
    "integer_comparison",
    "lexical_antonym",
    "category_completion",
    "world_capitals",
)
_FAMILY_PREFIXES = {
    "integer_addition": "addition",
    "integer_subtraction": "subtraction",
    "integer_comparison": "comparison",
    "lexical_antonym": "antonym",
    "category_completion": "category",
    "world_capitals": "capital",
}

MLP_MODE_IDS = tuple(f"mode.mutation.{2 * block + 2}" for block in range(18))
MLP_INVOCATION_IDS = tuple(f"mutation.{2 * block + 2}" for block in range(18))
PAIR_COUNT = 153
DIRECTED_OFF_DIAGONAL_COUNT = 306
FAMILY_DIRECTED_STAT_COUNT = 1_836
FAMILY_SELF_REPLAY_STAT_COUNT = 108
FAMILY_EXAMPLE_DIRECTED_STAT_COUNT = 18_360

PROBES_PER_MODE_EXAMPLE = 2
PROBES_PER_MODE_FAMILY = 20
PROBES_PER_MODE = 120
TOTAL_ORIGIN_PROBES = 2_160
CROSS_EXECUTION_COUNT = TOTAL_ORIGIN_PROBES * PINNED_LAYER_COUNT

RESIDUAL_PROFILE_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640)
BOOTSTRAP_SEED = 2_311
BOOTSTRAP_RESAMPLES = 10_000
STABLE_LOO_ADJACENCY_MINIMUM = 5
STABLE_BOOTSTRAP_ADJACENCY_FREQUENCY_MINIMUM = 0.80

ATLAS_CONFIG = {
    "energy_epsilon": 1.0e-12,
    "sufficient_stat_relative_tolerance": 1.0e-7,
    "connectivity_tolerance": 1.0e-12,
    "fiedler_eigengap_relative_tolerance": 1.0e-8,
    "fiedler_coordinate_tie_tolerance": 1.0e-10,
    "nearest_neighbor_tie_tolerance": 1.0e-10,
    "affinity_max_exponent": 700.0,
}

COMPARISON_ABI = {
    "abi_name": "carrier_mutation.token_local.h640",
    "abi_version": 1,
    "input_signature": "raw_carrier_row.float.h640",
    "output_signature": "additive_full_atom_delta.float.h640",
    "context_signature": "token_local.no_external_context",
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


def _example_ids(family_id: str) -> list[str]:
    prefix = _FAMILY_PREFIXES[family_id]
    return [f"{prefix}-{ordinal:02d}" for ordinal in range(1, 11)]


def _family_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for family_ordinal, family_id in enumerate(ORDERED_FAMILY_IDS):
        manifest: dict[str, object] = {
            "family_ordinal": family_ordinal,
            "family_id": family_id,
            "ordered_example_ids": _example_ids(family_id),
            "example_count": 10,
            "role": "spent_descriptive_discovery",
        }
        manifest["family_manifest_sha256"] = _sha256(
            _FAMILY_MANIFEST_HASH_DOMAIN,
            manifest,
        )
        manifests.append(manifest)
    return manifests


def _mode_catalog() -> list[dict[str, object]]:
    modes: list[dict[str, object]] = []
    for block_ordinal, (mode_id, invocation_id) in enumerate(
        zip(MLP_MODE_IDS, MLP_INVOCATION_IDS, strict=True)
    ):
        modes.append(
            {
                "mode_id": mode_id,
                "mode_abi_id": "abi.carrier_mutation.token_local.h640.v1",
                "source_invocation_id": invocation_id,
                "source_atom_id": f"atom.mlp.layer.{block_ordinal}",
                "source_block_ordinal": block_ordinal,
                "source_implementation_label": "MLP",
                "source_operator_label": "gated_GELU_MLP",
                "generic_core_mode_kind": (
                    "carrier_mutation.token_local.h640"
                ),
                "cross_execution_eligible": True,
                "native_source_state_sha256": (
                    "captured_and_authenticated_at_runtime"
                ),
                "provenance_fields_allowed_as_numerical_features": False,
            }
        )
    return modes


def _within_family_bootstrap_config() -> dict[str, object]:
    return {
        "draw_catalog_digest_encoding": (
            "sha256(catalog_domain||canonical_config||"
            "repeated(resample_u64be,family_u32be,draw_u32be,example_u32be))"
        ),
        "draw_catalog_digest_domain_hex": (
            _BOOTSTRAP_CATALOG_HASH_DOMAIN.hex()
        ),
        "draw_digest_selection": (
            "first_8_sha256_bytes_u64be_mod_example_count"
        ),
        "draw_preimage_encoding": (
            "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
        ),
        "draw_protocol": "sha256_counter_modulo_example_count_v1",
        "draw_protocol_domain_hex": _BOOTSTRAP_DRAW_HASH_DOMAIN.hex(),
        "draws_per_family": 10,
        "families": [
            {
                "family_id": family_id,
                "example_cluster_ids": _example_ids(family_id),
            }
            for family_id in ORDERED_FAMILY_IDS
        ],
        "maximum_degenerate_fraction": 0.01,
        "resample_count": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "sampling_unit": "example_plus_selected_probes_cluster",
        "with_replacement": True,
    }


def _atlas_config_payload() -> dict[str, object]:
    return {
        "affinity_kernel": "exp_neg_distance_over_positive_median_v1",
        "affinity_max_exponent": ATLAS_CONFIG["affinity_max_exponent"],
        "connectivity_tolerance": ATLAS_CONFIG["connectivity_tolerance"],
        "cosine_role": "diagnostic_only",
        "distance_metric": "bidirectional_equal_family_mean_relative_sse_v1",
        "energy_epsilon": ATLAS_CONFIG["energy_epsilon"],
        "fiedler_coordinate_tie_tolerance": ATLAS_CONFIG[
            "fiedler_coordinate_tie_tolerance"
        ],
        "fiedler_eigengap_relative_tolerance": ATLAS_CONFIG[
            "fiedler_eigengap_relative_tolerance"
        ],
        "laplacian": "symmetric_normalized_v1",
        "mode_id_role": "receipt_label_only",
        "mode_kind_role": "receipt_label_only",
        "nearest_neighbor_tie_tolerance": ATLAS_CONFIG[
            "nearest_neighbor_tie_tolerance"
        ],
        "ordering_quality_random_permutation_count": 10_000,
        "ordering_quality_random_permutation_seed": 0,
        "state_fingerprint_role": "canonical_axis_and_global_reversal_only",
        "sufficient_stat_relative_tolerance": ATLAS_CONFIG[
            "sufficient_stat_relative_tolerance"
        ],
    }


def _build_without_hash() -> dict[str, object]:
    family_manifests = _family_manifests()
    mode_catalog = _mode_catalog()
    return {
        "schema": GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_SCHEMA,
        "format_version": GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_FORMAT_VERSION,
        "artifact_role": (
            "setup_only_exact_no_compression_MLP_first_typed_mode_"
            "similarity_atlas"
        ),
        "setup_receipt": {
            "predecessor_artifacts_opened_by_builder": False,
            "model_weights_loaded": False,
            "prompt_panel_or_prompt_text_opened": False,
            "tokenization_performed": False,
            "carrier_or_atom_delta_captured": False,
            "output_gradient_or_Fisher_computed": False,
            "foreign_atom_cross_execution_performed": False,
            "distance_or_affinity_computed": False,
            "seriation_or_bootstrap_performed": False,
            "experiment_report_written": False,
            "new_experiment_metrics_present": False,
        },
        "predecessor_chain": {
            "setup": {
                "path": str(PINNED_PREDECESSOR_SETUP_PATH),
                "schema": (
                    "fisher_graph.gemma3_mlp_shared_family_residual_adapter."
                    "setup.v1"
                ),
                "setup_sha256": PINNED_PREDECESSOR_SETUP_SHA256,
                "file_sha256": PINNED_PREDECESSOR_SETUP_FILE_SHA256,
            },
            "report": {
                "path": str(PINNED_PREDECESSOR_REPORT_PATH),
                "schema": (
                    "fisher_graph.gemma3_mlp_shared_family_residual_adapter."
                    "experiment.v1"
                ),
                "report_sha256": PINNED_PREDECESSOR_REPORT_SHA256,
                "file_sha256": PINNED_PREDECESSOR_REPORT_FILE_SHA256,
            },
            "required_outcome": {
                "diagnostic_outcome": (
                    "shared_family_pair_mechanism_not_qualified"
                ),
                "dense_exact_control_passed_before_compressed_fit": True,
                "pair_mechanism_qualified": False,
                "B_guard_winner": None,
                "C_assessment_opened": False,
                "selected_pair_is_not_inherited_as_a_family": True,
            },
            "both_internal_and_file_SHA256_values_verified_before_model_load": (
                True
            ),
            "predecessor_pair_or_metric_may_seed_order_or_break_ties": False,
        },
        "model_contract": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "source_unique_stored_scalars": PINNED_MODEL_SCALARS,
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "layer_count": PINNED_LAYER_COUNT,
            "mutation_count": PINNED_MUTATION_COUNT,
            "event_count": PINNED_EVENT_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "parameter_state_and_training_mode_immutable": True,
            "full_native_model_used_only_for_exact_capture_and_gradients": True,
        },
        "mode_ontology": {
            "mode_definition": (
                "one_parameterized_state_mutation_operator_at_one_causal_"
                "occurrence_with_an_explicit_input_output_and_context_ABI"
            ),
            "implementation_names_are_provenance_not_ontology": True,
            "layer_block_invocation_and_component_labels_are_not_features": True,
            "abi_catalog": [
                {
                    "mode_abi_id": "abi.carrier_mutation.token_local.h640.v1",
                    "input": "one_exact_raw_carrier_row_float_h640",
                    "output": "one_additive_full_atom_delta_row_float_h640",
                    "context_dependency": "token_local",
                    "cross_execution_rule": (
                        "any_exact_atom_with_this_identical_ABI_may_be_"
                        "evaluated_on_any_origin_row"
                    ),
                    "eligible_in_this_setup": True,
                    "source_implementation_labels": ["MLP"],
                },
                {
                    "mode_abi_id": (
                        "abi.carrier_mutation.sequence_routed.h640.v1"
                    ),
                    "input": (
                        "exact_carrier_sequence_plus_mask_position_and_"
                        "routing_context"
                    ),
                    "output": "additive_carrier_sequence_delta_h640",
                    "context_dependency": "causal_sequence_routed",
                    "eligible_in_this_setup": False,
                    "source_implementation_labels": ["attention"],
                    "future_protocol_required": True,
                },
                {
                    "mode_abi_id": "abi.lexical_boundary.vocab_h640.v1",
                    "input": "token_identity_or_terminal_carrier_h640",
                    "output": "initial_carrier_h640_or_vocabulary_logits",
                    "context_dependency": "vocabulary_boundary",
                    "eligible_in_this_setup": False,
                    "source_implementation_labels": [
                        "embedding_ingress",
                        "tied_lexical_egress",
                    ],
                    "future_protocol_required": True,
                },
            ],
            "future_attention_and_lexical_atlases_extend_the_same_ontology": (
                True
            ),
            "cross_ABI_comparison_requires_a_preregistered_exact_adapter": True,
            "no_cross_ABI_adapter_is_present_here": True,
        },
        "scope_freeze": {
            "eligible_mode_abi_id": (
                "abi.carrier_mutation.token_local.h640.v1"
            ),
            "eligible_mode_count": PINNED_LAYER_COUNT,
            "ordered_mode_ids": list(MLP_MODE_IDS),
            "mode_catalog": mode_catalog,
            "unordered_pair_count": PAIR_COUNT,
            "directed_off_diagonal_count": DIRECTED_OFF_DIAGONAL_COUNT,
            "per_family_directed_cross_mode_stat_count": (
                FAMILY_DIRECTED_STAT_COUNT
            ),
            "per_family_self_replay_stat_count": FAMILY_SELF_REPLAY_STAT_COUNT,
            "per_family_example_directed_cross_mode_stat_count": (
                FAMILY_EXAMPLE_DIRECTED_STAT_COUNT
            ),
            "every_pair_and_both_directions_measured": True,
            "nonadjacency_filter_present": False,
            "source_block_distance_used_as_eligibility_or_feature": False,
            "source_parameters_or_persistent_buffers_changed": False,
            "candidate_model_constructed": False,
            "generator_shared_merged_deleted_or_replaced": False,
            "compression_or_low_rank_fit_performed": False,
            "carrier_edge_changed_or_learned": False,
            "attention_or_lexical_mode_measured": False,
        },
        "atlas_config": _atlas_config_payload(),
        "comparison_abi": dict(COMPARISON_ABI),
        "within_family_bootstrap_config": _within_family_bootstrap_config(),
        "discovery_data_firewall": {
            "panel_path": str(PINNED_PANEL_PATH),
            "panel_sha256": PINNED_PANEL_SHA256,
            "panel_id": PINNED_PANEL_ID,
            "family_count": 6,
            "example_count": 60,
            "ordered_family_ids": list(ORDERED_FAMILY_IDS),
            "family_manifests": family_manifests,
            "one_role_only": "spent_descriptive_discovery",
            "all_six_families_visible_to_descriptive_geometry": True,
            "leave_one_family_out_views_are_stability_analyses_not_holdouts": (
                True
            ),
            "all_60_examples_are_spent_after_this_atlas": True,
            "raw_prompt_text_or_token_ids_written_to_setup_or_report": False,
            "future_attention_or_lexical_descriptive_protocol_may_reuse_panel": (
                "only_if_it_makes_no_merge_compression_or_selection_claim"
            ),
            "every_future_merge_or_compression_protocol_requires": [
                "a_new_panel_with_a_distinct_authenticated_SHA256",
                "zero_example_id_overlap",
                "zero_exact_prompt_text_overlap",
                "zero_paraphrase_or_template_instance_overlap",
                "a_preregistered_overlap_audit_before_model_load",
                "no_atlas_metric_access_while_authoring_fresh_examples",
            ],
            "family_label_overlap_in_a_future_panel": (
                "allowed_only_if_preregistered_but_never_counts_as_fresh_"
                "prompt_evidence_by_itself"
            ),
            "this_atlas_may_not_choose_a_future_merge_rank_or_threshold": True,
        },
        "probe_sampling": {
            "capture_population": (
                "all_valid_nonpadding_causal_prompt_rows_for_each_origin_"
                "mode_and_panel_example"
            ),
            "valid_row_identity": (
                "family_id_example_id_and_zero_based_prompt_token_index_"
                "without_recording_token_identity"
            ),
            "Fisher_objective": (
                "sum_across_examples_of_gold_cross_entropy_under_the_native_"
                "restricted_four_way_softmax_at_the_last_valid_prompt_query"
            ),
            "choice_tokenization_gate": {
                "choice_encoding": "tokenize_one_leading_space_plus_choice",
                "required_token_count_per_choice": 1,
                "four_choice_token_ids_must_be_distinct_within_example": True,
                "query_position": "last_valid_nonpadding_prompt_position",
                "restricted_logits_order": "canonical_panel_choice_order",
                "gold_index": "canonical_panel_correct_choice",
                "loss_reduction_across_examples": "sum_not_mean",
                "token_ids_may_appear_in_setup_or_report": False,
                "per_example_choice_tokenization_and_query_receipt_SHA256_"
                "required": True,
            },
            "coordinatewise_output_Fisher": (
                "square_in_float64_of_the_source_dtype_gradient_of_that_"
                "objective_with_respect_to_each_exact_full_atom_delta_"
                "coordinate_at_every_causal_prompt_row"
            ),
            "row_energy": "sum_over_640_coordinates_of_output_Fisher",
            "selection_per_origin_mode_and_example": (
                "largest_row_energy_first_keep_exactly_two_distinct_rows"
            ),
            "ordered_tie_break": [
                "smaller_zero_based_token_index",
                "lexicographically_smaller_SHA256_of_canonical_float32_"
                "raw_carrier_delta_and_gradient_bytes",
                "fail_closed_if_row_identity_remains_ambiguous",
            ],
            "all_rows_seen_before_top_two_selection": True,
            "probes_per_origin_mode_example": PROBES_PER_MODE_EXAMPLE,
            "probes_per_origin_mode_family": PROBES_PER_MODE_FAMILY,
            "probes_per_origin_mode": PROBES_PER_MODE,
            "total_origin_probes": TOTAL_ORIGIN_PROBES,
            "cross_evaluate_all_18_exact_atoms_per_origin_probe": True,
            "full_atom_cross_executions": CROSS_EXECUTION_COUNT,
            "foreign_atom_cross_executions": TOTAL_ORIGIN_PROBES * 17,
            "origin_atom_self_replays": TOTAL_ORIGIN_PROBES,
            "self_replay_is_an_exact_runtime_gate_not_a_pair_distance": True,
            "self_replay_numerical_parity_gate": {
                "receipt_count": FAMILY_SELF_REPLAY_STAT_COUNT,
                "receipt_unit": "origin_mode_by_panel_family",
                "finite_values_required": True,
                "maximum_absolute_delta_error": 1.0e-6,
                "delta_NRMSE_maximum": 1.0e-7,
                "reason_not_bitwise": (
                    "capture_and_concatenated_atlas_replay_may_use_different_"
                    "GEMM_batch_shapes"
                ),
                "claim": "same_operator_float32_numerical_parity",
                "original_batch_shape_literal_hash_replay": (
                    "optional_diagnostic_reported_separately_if_executed"
                ),
            },
            "probe_selection_catalog_and_SHA256_required": True,
            "probe_catalog_hash_domain_hex": _PROBE_DRAW_HASH_DOMAIN.hex(),
            "insufficient_valid_rows_fails_closed": True,
        },
        "directed_distance_contract": {
            "direction_definition": (
                "i_to_j_uses_only_i_origin_exact_raw_carrier_probes_and_i_"
                "origin_output_Fisher"
            ),
            "reference": "Gi_of_xi_exact_full_atom_delta",
            "candidate": "Gj_of_xi_exact_full_atom_delta",
            "coordinate_weight": "g_i_squared_diagonal_output_Fisher",
            "per_family_sufficient_statistics": {
                "weighted_reference_energy": "sum(g_i^2*Gi(x_i)^2)",
                "weighted_candidate_energy": "sum(g_i^2*Gj(x_i)^2)",
                "weighted_dot": "sum(g_i^2*Gi(x_i)*Gj(x_i))",
                "weighted_residual_SSE": (
                    "reference_energy+candidate_energy-2*weighted_dot"
                ),
            },
            "per_family_relative_SSE": (
                "weighted_residual_SSE/max(weighted_reference_energy,"
                "atlas_config.energy_epsilon)"
            ),
            "reference_energy_must_be_finite_and_nonnegative": True,
            "relative_SSE_denominator": (
                "max(weighted_reference_energy,atlas_config.energy_epsilon)"
            ),
            "primary_directed_distance": (
                "arithmetic_mean_of_the_six_per_family_relative_SSE_values"
            ),
            "family_weighting": "equal_one_sixth_each_not_row_weighted",
            "primary_symmetric_distance": "0.5*(d(i_to_j)+d(j_to_i))",
            "diagonal_exactly_zero": True,
            "off_diagonal_must_be_finite_and_nonnegative": True,
            "operator_execution_dtype": "source_float32",
            "sufficient_statistic_accumulation_dtype": "float64",
            "negative_roundoff_within_float64_tolerance_clamped_to_zero": True,
            "distance_component_standardization_or_mixing_present": False,
        },
        "diagnostic_geometry": {
            "diagnostics_never_change_primary_distance_or_order": True,
            "weighted_bidirectional_cosine": {
                "directed_cosine": (
                    "weighted_dot/sqrt(max(weighted_reference_energy,"
                    "atlas_config.energy_epsilon)*max(weighted_candidate_"
                    "energy,atlas_config.energy_epsilon))"
                ),
                "symmetric_cosine_distance": (
                    "0.5*((1-cos_i_to_j)+(1-cos_j_to_i))"
                ),
                "denominator_uses_same_atlas_config_energy_epsilon": True,
            },
            "raw_unweighted_diagnostics": {
                "NRMSE": "sqrt(sum((Gi-Gj)^2)/sum(Gi^2))",
                "cosine": "sum(Gi*Gj)/sqrt(sum(Gi^2)*sum(Gj^2))",
                "aggregation": "per_family_then_equal_family_mean",
                "may_enter_primary_distance": False,
            },
            "weighted_residual_rank_profile": {
                "matrix_rows": (
                    "concatenate_elementwise_sqrt(g_i_squared)_times_"
                    "(Gi(x_i)-Gj(x_i))_for_the_120_i_origin_probes"
                ),
                "matrix_columns": PINNED_HIDDEN_WIDTH,
                "SVD_dtype": "float64",
                "rank_order": list(RESIDUAL_PROFILE_RANKS),
                "reported_value": (
                    "cumulative_squared_singular_value_energy_fraction"
                ),
                "report_both_directions_separately": True,
                "may_enter_primary_distance_or_seriation": False,
            },
        },
        "affinity_and_seriation": {
            "input": "full_18_by_18_primary_symmetric_distance_matrix",
            "distance_scale": (
                "median_of_all_153_strictly_positive_off_diagonal_distances"
            ),
            "zero_or_nonfinite_distance_scale_fails_closed": True,
            "dense_affinity": (
                "A_ij=exp(-min(D_ij/distance_scale,700))_for_i_not_equal_j;"
                "A_ii=0"
            ),
            "sparsification_or_nearest_neighbor_graph_used": False,
            "affinity_exponent_is_clipped_at_atlas_config_maximum": True,
            "laplacian": (
                "symmetric_normalized_I-D^(-1/2)*A*D^(-1/2)"
            ),
            "eigendecomposition": (
                "real_symmetric_float64_eigh_with_eigenvalues_ascending"
            ),
            "Fiedler_vector": "eigenvector_at_zero_based_index_1",
            "raw_order": "ascending_Fiedler_coordinate",
            "numerical_tolerances": "exactly_the_top_level_atlas_config",
            "fail_closed_conditions": [
                "Fiedler_eigenvalue_not_above_connectivity_tolerance",
                "lambda_3_minus_lambda_2_not_above_fiedler_eigengap_"
                "relative_tolerance_times_max(1,abs(lambda_2),abs(lambda_3))",
                "any_sorted_Fiedler_coordinate_gap_not_above_coordinate_tie_"
                "tolerance_times_max(1,max_abs_Fiedler_coordinate)",
                "any_mode_nearest_neighbor_gap_not_above_nearest_neighbor_"
                "tie_tolerance_times_max(1,first_distance,second_distance)",
                "any_nonfinite_matrix_eigenvalue_or_coordinate",
            ],
            "global_reversal_resolution": (
                "compare_the_forward_and_reversed_sequences_of_native_source_"
                "state_SHA256_values_and_keep_the_lexicographically_smaller_"
                "sequence"
            ),
            "state_fingerprint_role": (
                "canonical_input_axis_and_global_reversal_only"
            ),
            "state_fingerprint_never_enters_distance_affinity_or_numerical_"
            "tie_breaking": True,
            "mode_ID_invocation_block_ordinal_or_source_label_tie_break": False,
            "numerical_ties_are_never_broken_by_provenance": True,
            "ordering_quality_diagnostic": {
                "path_cost": "sum_adjacent_primary_symmetric_distances",
                "random_permutation_count": 10_000,
                "random_permutation_seed": 0,
                "random_generator": "numpy_PCG64",
                "may_change_order_or_hypotheses": False,
            },
        },
        "stability_contract": {
            "leave_one_family_out": {
                "fold_count": 6,
                "fold_order": list(ORDERED_FAMILY_IDS),
                "each_fold_distance": (
                    "equal_mean_of_the_five_included_family_relative_SSE_"
                    "matrices"
                ),
                "same_affinity_seriation_and_degeneracy_rules": True,
                "held_out_family_is_not_a_clean_assessment_role": True,
                "report_order_adjacency_set_and_full_order_adjacency_retention": (
                    True
                ),
            },
            "within_family_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "cluster_unit": "panel_example_with_its_two_selected_rows",
                "draw_rule": (
                    "for_each_family_and_resample_draw_10_example_ordinals_"
                    "with_replacement"
                ),
                "counter_PRNG": (
                    "uint64_big_endian_first_8_bytes_of_SHA256(domain||"
                    "seed_u64||resample_u64||family_u32||draw_u32)_mod_10"
                ),
                "counter_hash_domain_hex": _BOOTSTRAP_DRAW_HASH_DOMAIN.hex(),
                "family_ordinal_order": list(ORDERED_FAMILY_IDS),
                "same_drawn_example_indices_shared_by_all_modes_and_pairs": (
                    True
                ),
                "each_resample_recomputes_distance_affinity_and_Fiedler_order": (
                    True
                ),
                "degenerate_resample_count_reported_and_excluded_from_"
                "frequency_denominator": True,
                "fail_if_more_than_one_percent_resamples_degenerate": True,
                "report_unordered_adjacency_frequency_for_all_153_pairs": True,
                "bootstrap_draw_catalog_SHA256_required": True,
            },
            "stable_neighbor_hypothesis_rule": {
                "must_be_adjacent_in_full_order": True,
                "leave_one_family_out_adjacency_count_minimum": (
                    STABLE_LOO_ADJACENCY_MINIMUM
                ),
                "bootstrap_adjacency_frequency_minimum": (
                    STABLE_BOOTSTRAP_ADJACENCY_FREQUENCY_MINIMUM
                ),
                "result_label": "stable_neighbor_hypothesis_only",
                "does_not_establish_a_generator_family_or_merge": True,
            },
            "absence_of_stable_neighbors_is_a_valid_result": True,
        },
        "execution_and_report_contract": {
            "preflight_before_model_load": [
                "authenticate_this_setup_internal_and_file_SHA256",
                "authenticate_predecessor_setup_internal_and_file_SHA256",
                "authenticate_predecessor_report_internal_and_file_SHA256",
                "authenticate_panel_file_SHA256_family_and_example_manifests",
                "refuse_an_existing_output_path",
            ],
            "exact_source_state_fingerprint_before_and_after_equal": True,
            "deterministic_replay_of_probe_selection_distances_and_order": True,
            "required_report_fields": [
                "setup_model_predecessor_and_panel_provenance",
                "all_18_runtime_source_state_SHA256_values",
                "probe_selection_catalog_and_SHA256",
                "choice_tokenization_and_last_query_receipt_SHA256",
                "per_example_per_family_sufficient_statistic_SHA256_values",
                "six_directed_family_distance_matrices",
                "108_per_family_mode_self_replay_numerical_parity_receipts",
                "full_directed_and_symmetric_distance_matrices",
                "weighted_and_raw_diagnostic_matrices",
                "bidirectional_weighted_residual_rank_profiles",
                "distance_scale_affinity_laplacian_eigenvalues_and_order",
                "six_leave_one_family_out_orders_and_adjacency_receipts",
                "bootstrap_draw_SHA256_degeneracy_count_and_adjacency_frequencies",
                "stable_neighbor_hypotheses_or_an_authenticated_empty_list",
                "runtime_peak_memory_and_source_call_audit",
            ],
            "report_contains_model_weights_prompt_text_or_token_ids": False,
            "report_schema": (
                "fisher_graph.gemma3_mlp_mode_similarity_atlas.experiment.v1"
            ),
        },
        "claim_boundary": {
            "setup_only": True,
            "model_prompt_or_metric_access_occurred": False,
            "descriptive_exact_geometry_only": True,
            "whole_model_is_executed_for_capture_but_not_modified": True,
            "compression_parameter_MAC_byte_memory_or_latency_reduction_claimed": (
                False
            ),
            "shared_generator_family_or_merge_claimed": False,
            "causal_or_functional_interchangeability_claimed": False,
            "edge_similarity_or_edge_compression_claimed": False,
            "attention_or_lexical_geometry_claimed": False,
            "stable_neighbors_are_hypotheses_requiring_fresh_data": True,
            "a_future_negative_result_may_report_no_qualifying_neighbors": True,
        },
    }


def _validate_built_setup(value: Mapping[str, object]) -> None:
    scope = value["scope_freeze"]
    if scope["ordered_mode_ids"] != list(MLP_MODE_IDS):
        raise ValueError("MLP mode order drifted")
    if len(scope["mode_catalog"]) != PINNED_LAYER_COUNT:
        raise ValueError("MLP mode catalog count drifted")
    if len({mode["mode_id"] for mode in scope["mode_catalog"]}) != 18:
        raise ValueError("MLP mode ids are not unique")
    if any(
        mode["mode_abi_id"] != "abi.carrier_mutation.token_local.h640.v1"
        for mode in scope["mode_catalog"]
    ):
        raise ValueError("an ineligible ABI entered the MLP atlas")
    if value["atlas_config"] != _atlas_config_payload():
        raise ValueError("generic atlas numerical config drifted")
    if value["comparison_abi"] != COMPARISON_ABI:
        raise ValueError("generic atlas comparison ABI drifted")
    if value["within_family_bootstrap_config"] != (
        _within_family_bootstrap_config()
    ):
        raise ValueError("within-family bootstrap config drifted")

    firewall = value["discovery_data_firewall"]
    manifests = firewall["family_manifests"]
    if [manifest["family_id"] for manifest in manifests] != list(
        ORDERED_FAMILY_IDS
    ):
        raise ValueError("discovery family order drifted")
    examples = [
        example_id
        for manifest in manifests
        for example_id in manifest["ordered_example_ids"]
    ]
    if len(examples) != 60 or len(set(examples)) != 60:
        raise ValueError("discovery examples are not disjoint and exhaustive")

    probes = value["probe_sampling"]
    if probes["total_origin_probes"] != 18 * 60 * 2:
        raise ValueError("origin probe count drifted")
    if probes["full_atom_cross_executions"] != 18 * 60 * 2 * 18:
        raise ValueError("cross-execution count drifted")

    distance = value["directed_distance_contract"]
    if distance["primary_symmetric_distance"] != (
        "0.5*(d(i_to_j)+d(j_to_i))"
    ):
        raise ValueError("symmetric distance formula drifted")
    if distance["distance_component_standardization_or_mixing_present"]:
        raise ValueError("diagnostics may not enter primary distance")
    if value["scope_freeze"]["carrier_edge_changed_or_learned"]:
        raise ValueError("edge changes are forbidden")
    if value["claim_boundary"]["shared_generator_family_or_merge_claimed"]:
        raise ValueError("the atlas may not claim a family")


def build_gemma3_mlp_mode_similarity_atlas_setup() -> dict[str, object]:
    setup = _build_without_hash()
    _validate_built_setup(setup)
    setup["setup_sha256"] = _setup_sha256(setup)
    return setup


def validate_gemma3_mlp_mode_similarity_atlas_setup(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("MLP mode-similarity atlas setup must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if materialized != build_gemma3_mlp_mode_similarity_atlas_setup():
        raise ValueError("MLP mode-similarity atlas setup is not canonical")


def load_gemma3_mlp_mode_similarity_atlas_setup(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_mlp_mode_similarity_atlas_setup(value)
    return value


def write_gemma3_mlp_mode_similarity_atlas_setup(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(setup, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite an MLP mode-similarity atlas setup"
        ) from error
    return setup


@dataclass(frozen=True, slots=True)
class Gemma3MLPModeSimilarityAtlasSetup:
    """Typed wrapper around the immutable no-compression atlas setup."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_mlp_mode_similarity_atlas_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3MLPModeSimilarityAtlasSetup":
        return cls(build_gemma3_mlp_mode_similarity_atlas_setup())

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
        load_gemma3_mlp_mode_similarity_atlas_setup(args.output)
    else:
        write_gemma3_mlp_mode_similarity_atlas_setup(args.output)
    return 0


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "COMPARISON_ABI",
    "CROSS_EXECUTION_COUNT",
    "DEFAULT_OUTPUT",
    "DIRECTED_OFF_DIAGONAL_COUNT",
    "FAMILY_DIRECTED_STAT_COUNT",
    "FAMILY_EXAMPLE_DIRECTED_STAT_COUNT",
    "FAMILY_SELF_REPLAY_STAT_COUNT",
    "ATLAS_CONFIG",
    "GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_FORMAT_VERSION",
    "GEMMA3_MLP_MODE_SIMILARITY_ATLAS_SETUP_SCHEMA",
    "Gemma3MLPModeSimilarityAtlasSetup",
    "MLP_INVOCATION_IDS",
    "MLP_MODE_IDS",
    "ORDERED_FAMILY_IDS",
    "PAIR_COUNT",
    "PINNED_MODEL_ID",
    "PINNED_MODEL_REVISION",
    "PINNED_PANEL_SHA256",
    "PROBES_PER_MODE_EXAMPLE",
    "RESIDUAL_PROFILE_RANKS",
    "STABLE_BOOTSTRAP_ADJACENCY_FREQUENCY_MINIMUM",
    "STABLE_LOO_ADJACENCY_MINIMUM",
    "TOTAL_ORIGIN_PROBES",
    "build_gemma3_mlp_mode_similarity_atlas_setup",
    "load_gemma3_mlp_mode_similarity_atlas_setup",
    "main",
    "validate_gemma3_mlp_mode_similarity_atlas_setup",
    "write_gemma3_mlp_mode_similarity_atlas_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
