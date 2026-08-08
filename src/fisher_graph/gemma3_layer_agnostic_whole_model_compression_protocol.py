"""Preregister the layer-agnostic Gemma whole-model compression rung.

This file is intentionally standard-library only.  Building or writing the
protocol cannot import Torch, Transformers, load a checkpoint, tokenize a
prompt, fit a coefficient, or execute an experiment.

The protocol removes native *layers* from the proposed runtime ontology while
preserving causal mutation occurrences.  Reusable generator families may span
any source layers, but their invocations remain separate ordered graph nodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_layer_agnostic_whole_model_compression.protocol.v1"
)
LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_FORMAT_VERSION = 1
DEFAULT_PROTOCOL_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-layer-agnostic-whole-model-compression-protocol-v1.json"
)

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_LAYER_COUNT = 18
PINNED_HIDDEN_WIDTH = 640
PINNED_MLP_WIDTH = 2048
PINNED_VOCABULARY_SIZE = 262_144
PINNED_SOURCE_UNIQUE_PARAMETER_COUNT = 268_098_176
PINNED_TIED_LEXICAL_PARAMETER_COUNT = 167_772_160
PINNED_TRANSFORMER_BODY_PARAMETER_COUNT = 100_325_376
PINNED_FINAL_RMSNORM_PARAMETER_COUNT = 640

_PROTOCOL_HASH_DOMAIN = (
    b"fisher_graph:gemma3-layer-agnostic-whole-model-compression:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(
        _PROTOCOL_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _json_safe(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _mutation_schedule() -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for layer_ordinal in range(PINNED_LAYER_COUNT):
        for local_stage in ("attention", "feed_forward"):
            mutation_index = len(schedule) + 1
            attention_kind = (
                "global"
                if layer_ordinal in (5, 11, 17)
                else "sliding_512"
            )
            schedule.append(
                {
                    "invocation_id": f"mutation.{mutation_index}",
                    "mutation_index": mutation_index,
                    "operator_contract": (
                        "causal_prefix_mixer"
                        if local_stage == "attention"
                        else "token_local"
                    ),
                    "attention_contract": (
                        attention_kind if local_stage == "attention" else None
                    ),
                    "provenance_only": {
                        "source_layer_ordinal": layer_ordinal,
                        "source_stage": local_stage,
                        "source_mutation_id": (
                            f"layer.{layer_ordinal}.{local_stage}"
                        ),
                    },
                }
            )
    return schedule


def _lexical_factor_scalar_count(rank: int) -> int:
    if rank == PINNED_HIDDEN_WIDTH:
        return PINNED_TIED_LEXICAL_PARAMETER_COUNT
    return rank * (PINNED_VOCABULARY_SIZE + PINNED_HIDDEN_WIDTH)


@dataclass(frozen=True, slots=True)
class LayerAgnosticWholeModelCompressionProtocol:
    """Frozen search space and scientific firewall for the future worker."""

    residual_rank_ladder: tuple[int, ...] = (640, 576, 512, 448, 384)
    lexical_rank_ladder: tuple[int, ...] = (640, 512, 384, 256)
    atlas_region_count_ladder: tuple[int, ...] = (1, 4, 8, 16)
    local_chart_correction_rank_ladder: tuple[int, ...] = (0, 8, 16, 32)
    generator_community_budget_ratio_ladder: tuple[float, ...] = (
        1.0,
        0.75,
        0.50,
        0.35,
    )
    maximum_incoming_transport_ladder: tuple[int, ...] = (1, 2, 4)
    active_chart_count_ladder: tuple[int, ...] = (1, 2)
    relative_ridge_ladder: tuple[float, ...] = (
        1.0e-6,
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
    )
    chart_temperature_ladder: tuple[float, ...] = (0.5, 1.0, 2.0)
    block_coordinate_sweeps: int = 8

    def __post_init__(self) -> None:
        self._require_descending_rank_ladder(
            self.residual_rank_ladder,
            maximum=PINNED_HIDDEN_WIDTH,
            label="residual_rank_ladder",
        )
        self._require_descending_rank_ladder(
            self.lexical_rank_ladder,
            maximum=PINNED_HIDDEN_WIDTH,
            label="lexical_rank_ladder",
        )
        for label, values, allow_zero in (
            ("atlas_region_count_ladder", self.atlas_region_count_ladder, False),
            (
                "local_chart_correction_rank_ladder",
                self.local_chart_correction_rank_ladder,
                True,
            ),
            (
                "maximum_incoming_transport_ladder",
                self.maximum_incoming_transport_ladder,
                False,
            ),
            ("active_chart_count_ladder", self.active_chart_count_ladder, False),
        ):
            if (
                not values
                or values != tuple(sorted(set(values)))
                or any(
                    type(value) is not int
                    or value < (0 if allow_zero else 1)
                    for value in values
                )
            ):
                raise ValueError(f"{label} is invalid")
        budgets = self.generator_community_budget_ratio_ladder
        if (
            not budgets
            or len(set(budgets)) != len(budgets)
            or any(
                not isinstance(value, float)
                or not math.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in budgets
            )
            or budgets != tuple(sorted(budgets, reverse=True))
        ):
            raise ValueError("generator community budget ladder is invalid")
        for label, values in (
            ("relative_ridge_ladder", self.relative_ridge_ladder),
            ("chart_temperature_ladder", self.chart_temperature_ladder),
        ):
            if (
                not values
                or values != tuple(sorted(set(values)))
                or any(
                    not isinstance(value, float)
                    or not math.isfinite(value)
                    or value <= 0.0
                    for value in values
                )
            ):
                raise ValueError(f"{label} is invalid")
        if type(self.block_coordinate_sweeps) is not int or (
            self.block_coordinate_sweeps <= 0
        ):
            raise ValueError("block_coordinate_sweeps must be positive")

    @staticmethod
    def _require_descending_rank_ladder(
        values: tuple[int, ...],
        *,
        maximum: int,
        label: str,
    ) -> None:
        if (
            not values
            or values != tuple(sorted(set(values), reverse=True))
            or values[0] != maximum
            or any(type(value) is not int or value <= 0 for value in values)
        ):
            raise ValueError(f"{label} is invalid")

    def state_dict(self) -> dict[str, object]:
        mutation_schedule = _mutation_schedule()
        if len(mutation_schedule) != 2 * PINNED_LAYER_COUNT:
            raise AssertionError("pinned mutation schedule is incomplete")
        lexical_curve = [
            {
                "rank": rank,
                "stored_scalars": _lexical_factor_scalar_count(rank),
                "source_scalar_reduction": (
                    PINNED_TIED_LEXICAL_PARAMETER_COUNT
                    - _lexical_factor_scalar_count(rank)
                ),
                "rank_640_is_exact_copied_tied_matrix": (
                    rank == PINNED_HIDDEN_WIDTH
                ),
            }
            for rank in self.lexical_rank_ladder
        ]
        arms = [
            {
                "arm_id": "native",
                "purpose": "teacher_and_resource_reference",
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "exact_whole_model_graph",
                "purpose": (
                    "standalone_exact_control_with_zero_live_teacher_calls"
                ),
                "copied_tied_lexical_matrix": True,
                "contains_copied_checkpoint_coefficients": True,
                "source_free_artifact": False,
                "exact_ordered_mutations": 36,
                "exact_terminal_rmsnorm_and_softcap": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "full_rank_layer_agnostic_identity",
                "purpose": "authenticate_chart_and_transport_machinery",
                "full_rank": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "global_residual_basis_control",
                "shared_residual_basis": True,
                "local_atlas": False,
                "cross_invocation_family_sharing": False,
                "compression_credit_allowed_after_guard": True,
            },
            {
                "arm_id": "boundary_local_basis_control",
                "basis_per_mutation_boundary": True,
                "adjacent_transports_only": True,
                "native_layer_objects_in_runtime": False,
                "compression_credit_allowed_after_guard": True,
            },
            {
                "arm_id": "budget_matched_modular_factorization",
                "invocation_local_factors": True,
                "cross_provenance_sharing": False,
                "compression_credit_allowed_after_guard": True,
            },
            {
                "arm_id": "unrestricted_layer_agnostic_primary",
                "pooled_charts": True,
                "cross_provenance_generator_families": True,
                "forward_skip_fanout_and_fanin": True,
                "shared_tied_lexical_factor": True,
                "compression_credit_allowed_after_guard": True,
            },
            {
                "arm_id": "matched_naive_svd",
                "serialized_bytes_must_not_exceed_primary": True,
                "byte_matching_precedes_mac_proximity": True,
                "logical_macs_reported_not_forced_equal": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "matched_fisher_deletion",
                "same_physical_budget_as_primary": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "primary_zero_transport_ablation",
                "primary_transport_weights_zeroed_without_refit": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "primary_family_shuffled_negative_control",
                "fit_rows_only_are_shuffled": True,
                "evaluation_rows_are_never_shuffled": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "exact_lexical_compressed_body_ablation",
                "exact_tied_lexical_matrix": True,
                "compressed_body": True,
                "compression_credit_allowed_after_guard": True,
            },
            {
                "arm_id": "compressed_lexical_exact_body_ablation",
                "compressed_tied_lexical_factor": True,
                "exact_body": True,
                "compression_credit_allowed_after_guard": True,
            },
        ]
        state: dict[str, object] = {
            "schema": (
                LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_SCHEMA
            ),
            "format_version": (
                LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_FORMAT_VERSION
            ),
            "artifact_role": "preregistered_setup_only",
            "setup_receipt": {
                "model_weights_loaded": False,
                "prompt_data_opened": False,
                "tokenization_performed": False,
                "fitting_performed": False,
                "experiment_executed": False,
                "metrics_present": False,
                "compression_success_claimed": False,
            },
            "pinned_source": {
                "model_id": PINNED_MODEL_ID,
                "model_revision": PINNED_MODEL_REVISION,
                "tokenizer_revision": PINNED_MODEL_REVISION,
                "runtime": "cpu_float32_eager_cache_free_prefill",
                "valid_query_rows_only": True,
                "decode_and_kv_cache": "deferred_to_future_protocol",
                "layer_count_as_provenance": PINNED_LAYER_COUNT,
                "hidden_width": PINNED_HIDDEN_WIDTH,
                "mlp_width": PINNED_MLP_WIDTH,
                "vocabulary_size": PINNED_VOCABULARY_SIZE,
            },
            "source_parameter_inventory": {
                "whole_model_unique_parameters": (
                    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
                "tied_embedding_and_head_counted_once": (
                    PINNED_TIED_LEXICAL_PARAMETER_COUNT
                ),
                "transformer_body": (
                    PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
                ),
                "final_rmsnorm": PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
                "inventory_closes_exactly": (
                    PINNED_TIED_LEXICAL_PARAMETER_COUNT
                    + PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
                    + PINNED_FINAL_RMSNORM_PARAMETER_COUNT
                    == PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
            },
            "runtime_ontology": {
                "native_layers_are_execution_objects": False,
                "native_layer_ids_retained_as_provenance_only": True,
                "generator_family_is_parameter_identity": True,
                "generator_invocation_is_temporal_identity": True,
                "same_family_may_have_multiple_invocations": True,
                "same_family_invocations_are_not_self_edges": True,
                "causal_order_key": "mutation_index",
                "layer_adjacency_is_edge_eligibility_feature": False,
                "skip_fanout_and_fanin_edges_allowed": True,
                "attention_cross_token_contract_is_preserved": True,
            },
            "whole_model_boundary_graph": {
                "lexical_ingress": {
                    "invocation_id": "lexical.ingress",
                    "mutation_index": 0,
                    "operator_contract": "vocabulary_boundary",
                },
                "mutation_invocations": mutation_schedule,
                "attention_runtime_contracts": {
                    "shared": {
                        "query_heads": 4,
                        "key_value_heads": 1,
                        "head_dimension": 256,
                        "query_pre_attention_scalar": 256,
                        "query_and_key_rmsnorm": True,
                        "grouped_query_attention": True,
                        "attention_softmax_dtype": "float32",
                        "attention_logit_softcap": None,
                        "causal_mask_required": True,
                        "rope_scaling": None,
                    },
                    "sliding_512": {
                        "window": 512,
                        "rope_theta": 10_000.0,
                    },
                    "global": {
                        "window": None,
                        "rope_theta": 1_000_000.0,
                    },
                    "global_source_layer_ordinals": [5, 11, 17],
                },
                "terminal_rmsnorm": {
                    "invocation_id": "terminal.rmsnorm",
                    "mutation_index": 37,
                    "compiled_exact_in_v1": True,
                },
                "lexical_egress": {
                    "invocation_id": "lexical.egress",
                    "mutation_index": 38,
                    "operator_contract": "vocabulary_boundary",
                    "logit_softcap_compiled_exact": True,
                },
                "expected_ordered_mutation_count": 36,
                "source_mutation_id_scheme": (
                    "layer.N.attention_then_layer.N.feed_forward"
                ),
                "exact_body_execution_fingerprint_required": True,
                "exact_carrier_session_scope": "body_mutations_only",
                "lexical_and_terminal_boundaries_outside_body_session": True,
                "residual_state_wire_is_first_class": True,
                "dense_residual_carrier_is_exact_control_not_compression": True,
                "candidate_requires_carrier_native_chart_mutations": True,
                "inputs_embeds_bypass_policy": (
                    "unsupported_in_v1_fail_closed_no_native_bypass"
                ),
                "fitted_plan_required_assembly_indexes": "0_through_38",
                "fitted_plan_protocol_hash_must_equal_this_artifact": True,
            },
            "layer_agnostic_discovery_policy": {
                "candidate_pair_scope": "any_shape_compatible_source_modes",
                "allowed_feature_families": [
                    "activation_geometry",
                    "fisher_score_factors",
                    "jacobian_and_jvp_fingerprints",
                    "finite_displacement_flow",
                    "operator_shape_and_token_dependency_contract",
                ],
                "forbidden_clustering_and_routing_features": [
                    "layer_id",
                    "layer_ordinal",
                    "layer_distance",
                    "same_layer",
                    "adjacent_layer",
                    "source_boundary_label",
                ],
                "provenance_label_permutation_must_preserve_assignments": True,
                "provenance_is_still_authenticated_in_artifact_hash": True,
                "raw_correlation_alone_establishes_dependency": False,
                "dependency_confirmation": (
                    "partial_residual_gain_then_held_direction_jvp_then_"
                    "finite_displacement"
                ),
            },
            "shared_lexical_boundary": {
                "shared_factor_id": "lexical.shared_factor",
                "ingress_and_egress_reference_same_factor": True,
                "physical_storage_multiplicity": 1,
                "physical_alias_audit_required": True,
                "equal_but_distinct_copies_count_as_tied": False,
                "factor_storage_layout": (
                    "single_packed_C_then_D_backing_span"
                ),
                "physical_span_receipt_covers_C_and_D": True,
                "separate_C_and_D_storage_allowed_in_v1": False,
                "factorization": "E_approximately_C_times_D",
                "ingress": "candidate_embedding_lookup_uses_shared_C_and_D",
                "egress": "candidate_logits_use_D_transpose_and_C_transpose",
                "full_vocabulary_retained": True,
                "adapter_embedding_semantics_and_scaling_preserved_exactly": True,
                "terminal_rmsnorm_preserved_exactly": True,
                "decoder_orientation_preserved_exactly": True,
                "logit_softcap_preserved_exactly": True,
                "arbitrary_basis_folding_across_nonlinearity_is_exact": False,
                "lexical_rank_resource_curve": lexical_curve,
            },
            "frozen_ladders": {
                "residual_rank": list(self.residual_rank_ladder),
                "lexical_rank": list(self.lexical_rank_ladder),
                "atlas_region_count": list(self.atlas_region_count_ladder),
                "local_chart_correction_rank": list(
                    self.local_chart_correction_rank_ladder
                ),
                "generator_community_budget_ratio": list(
                    self.generator_community_budget_ratio_ladder
                ),
                "maximum_incoming_transport_edges_per_invocation": list(
                    self.maximum_incoming_transport_ladder
                ),
                "active_charts_per_token": list(
                    self.active_chart_count_ladder
                ),
                "relative_ridge": list(self.relative_ridge_ladder),
                "chart_temperature": list(self.chart_temperature_ladder),
                "pilot_point": {
                    "residual_rank": 576,
                    "lexical_rank": 512,
                },
                "full_cartesian_product_exhausted": False,
                "ordered_resource_points": [
                    {
                        "resource_point_id": "conservative",
                        "residual_rank": 576,
                        "lexical_rank": 512,
                        "atlas_regions": 4,
                        "local_chart_correction_rank": 8,
                        "generator_community_budget_ratio": 0.75,
                        "maximum_incoming_transports": 1,
                        "active_charts_per_token": 1,
                        "relative_ridge": 1.0e-3,
                        "chart_temperature": 1.0,
                    },
                    {
                        "resource_point_id": "balanced",
                        "residual_rank": 512,
                        "lexical_rank": 384,
                        "atlas_regions": 8,
                        "local_chart_correction_rank": 16,
                        "generator_community_budget_ratio": 0.50,
                        "maximum_incoming_transports": 2,
                        "active_charts_per_token": 1,
                        "relative_ridge": 1.0e-3,
                        "chart_temperature": 1.0,
                    },
                    {
                        "resource_point_id": "compact",
                        "residual_rank": 448,
                        "lexical_rank": 384,
                        "atlas_regions": 8,
                        "local_chart_correction_rank": 16,
                        "generator_community_budget_ratio": 0.50,
                        "maximum_incoming_transports": 2,
                        "active_charts_per_token": 1,
                        "relative_ridge": 1.0e-3,
                        "chart_temperature": 1.0,
                    },
                    {
                        "resource_point_id": "aggressive",
                        "residual_rank": 384,
                        "lexical_rank": 256,
                        "atlas_regions": 16,
                        "local_chart_correction_rank": 32,
                        "generator_community_budget_ratio": 0.35,
                        "maximum_incoming_transports": 4,
                        "active_charts_per_token": 2,
                        "relative_ridge": 1.0e-2,
                        "chart_temperature": 1.0,
                    },
                ],
                "every_resource_point_is_fitted_before_selection": True,
                "resource_shortfall_stops_lower_ranks": False,
                "fidelity_monotonicity_assumed": False,
            },
            "frozen_numerics": {
                "deterministic_seed": 1729,
                "moment_accumulation_dtype": "float64",
                "ridge_solve_dtype": "float64",
                "runtime_materialization_dtype": "float32",
                "weighted_centering": True,
                "relative_ridge_definition": (
                    "lambda_times_mean_diagonal_of_weighted_gram"
                ),
                "canonical_svd_sign_rule": (
                    "largest_absolute_decoder_entry_is_positive"
                ),
                "block_coordinate_sweeps": self.block_coordinate_sweeps,
                "block_objective_must_be_monotone": True,
                "tie_break_order": [
                    "best_frozen_selection_objective",
                    "fewer_serialized_bytes",
                    "fewer_logical_macs",
                    "lower_region_count",
                    "lower_rank",
                    "canonical_family_and_invocation_id",
                ],
            },
            "frozen_fit_procedure": {
                "teacher_objective": (
                    "per_sequence_sum_of_valid_causal_lm_nll_before_"
                    "cross_sequence_mean"
                ),
                "causal_shift": "query_t_predicts_valid_token_t_plus_1",
                "ignore_padding_and_invalid_queries": True,
                "fisher_row_weight": (
                    "mean_over_mode_of_squared_mode_state_times_score_gradient"
                ),
                "fisher_weights_normalized_to_mean_one": True,
                "capture_sites": [
                    "adapter_pinned_lexical_embedding_output",
                    "pre_and_post_each_of_36_source_mutations",
                    "pre_terminal_rmsnorm",
                    "post_terminal_rmsnorm",
                    "pre_softcap_full_vocabulary_logits",
                    "post_softcap_full_vocabulary_logits",
                ],
                "row_identity": (
                    "prompt_family_id_example_id_logical_token_position"
                ),
                "capture_row_identity_hash_required_at_every_site": True,
                "primitive_discovery": (
                    "joint_shape_compatible_fisher_geometry_jacobian_"
                    "clustering_across_all_sites"
                ),
                "mode_id_is_a_clustering_feature": False,
                "topology_candidate_scope": "all_strictly_forward_invocations",
                "topology_score": (
                    "partial_fisher_weighted_residual_gain_beyond_local_"
                    "generator"
                ),
                "edge_shortlist_limit": (
                    "resource_point_maximum_incoming_transports_per_target"
                ),
                "edge_tie_break": (
                    "descending_partial_gain_then_source_then_target_"
                    "canonical_id"
                ),
                "dependency_confirmation": (
                    "held_direction_jvp_then_finite_displacement"
                ),
                "dependency_confirmation_rows": (
                    "A_fit_inner_direction_holdout_disjoint_from_"
                    "coefficient_fit_rows"
                ),
                "dependency_confirmation_may_open_A_select_B_or_C": False,
                "shared_family_strategy": (
                    "jointly_fit_each_family_bank_over_all_member_"
                    "invocations_then_freeze_before_any_progressive_fit"
                ),
                "progressive_order": [
                    "fit_shared_tied_lexical_factor_and_ingress_map",
                    "jointly_fit_and_freeze_all_shared_family_banks",
                    "fit_invocation_local_adapters_and_transports_in_"
                    "mutation_order_on_actual_candidate_trajectory",
                    "fit_exact_terminal_rmsnorm_boundary_adapter",
                    "fit_egress_map_using_the_same_tied_lexical_factor",
                    "recapture_complete_candidate_trajectory_then_run_"
                    "canonical_forward_adapter_transport_sweep",
                ],
                "global_family_banks_updated_after_freeze": False,
                "full_candidate_trajectory_recaptured_each_sweep": True,
                "block_coordinate_sweep_count": self.block_coordinate_sweeps,
                "monotone_fit_objective_required": True,
                "source_parameters_gradients_and_training_mode_immutable": True,
            },
            "arms": arms,
            "fit_and_validation_firewall": {
                "split_units": "prompt_or_paraphrase_family_is_atomic",
                "required_manifests": [
                    "A_fit",
                    "A_select",
                    "B_guard",
                    "C_assessment",
                ],
                "manifest_sha256_values": "required_at_execution",
                "row_prompt_and_family_overlap_allowed": False,
                "A_fit": (
                    "capture_all_boundaries_and_fit_features_coefficients_"
                    "communities_charts_and_topology"
                ),
                "A_fit_inner_direction_holdout": (
                    "held_direction_jvp_and_finite_displacement_only"
                ),
                "A_fit_inner_holdout_disjoint_from_coefficient_fit_rows": True,
                "A_fit_inner_holdout_may_touch_A_select_B_or_C": False,
                "fit_uses_actual_frozen_candidate_prefix_trajectory": True,
                "candidate_only_block_coordinate_refinement": True,
                "source_model_mutated": False,
                "A_select": (
                    "select_only_among_already_fitted_frozen_artifacts"
                ),
                "A_select_may_refit_coefficients_or_topology": False,
                "A_select_nuisance_choice_rule": (
                    "each_ordered_resource_point_has_one_frozen_ridge_"
                    "temperature_and_topology_budget"
                ),
                "pareto_frontier_frozen_before_B_guard": True,
                "B_guard_opened_once": True,
                "B_guard_may_refit_or_change_topology": False,
                "B_guard_scores_entire_frozen_frontier_once": True,
                "B_guard_multiplicity_control": (
                    "simultaneous_family_clustered_bootstrap_intervals"
                ),
                "B_guard_selection_rule": (
                    "highest_compression_resource_point_passing_all_"
                    "hard_fidelity_causality_and_accounting_gates"
                ),
                "B_guard_resource_failure_stops_frontier": False,
                "C_assessment_opened_only_after_guard_pass": True,
                "C_assessment_candidate_count": 1,
                "C_assessment_opened_once": True,
                "C_assessment_may_refit_select_or_change_thresholds": False,
                "failed_guard_requires_new_protocol_and_fresh_splits": True,
                "existing_a5e_prompt_split_is_heldout_confirmation": False,
                "existing_a5e_panel_role": "development_smoke_only",
            },
            "causal_and_runtime_invariants": {
                "edge_source_mutation_index_strictly_less_than_target": True,
                "self_reverse_and_cyclic_edges_forbidden": True,
                "future_token_edges_forbidden": True,
                "same_token_edges_do_not_mix_examples_or_positions": True,
                "causal_prefix_edges_require_source_position_lte_target": True,
                "same_sequence_and_segment_required": True,
                "sliding_global_mask_and_rope_semantics_preserved": True,
                "future_token_perturbation_prefix_invariance_required": True,
                "padding_batch_and_segment_invariance_required": True,
                "invalid_rows_inert": True,
                "deterministic_replay_required": True,
                "ordered_residual_mutation_closure_required": True,
                "skipped_mutation_requires_explicit_validated_replacement": True,
                "unsupported_chart_state_may_call_native_fallback": False,
                "zero_live_source_calls_required": True,
                "source_candidate_storage_disjoint_except_candidate_internal_tie": (
                    True
                ),
                "source_fingerprint_gradients_and_training_mode_unchanged": True,
            },
            "required_metrics": {
                "external": [
                    "nll_per_token_and_delta",
                    "perplexity_ratio",
                    "native_to_candidate_kl_per_token",
                    "top1_agreement_and_topk_overlap",
                    "logit_nrmse_max_error_and_cosine",
                    "downstream_accuracy_retained",
                    "worst_family_metrics",
                    "family_clustered_bootstrap_intervals",
                ],
                "internal": [
                    "per_boundary_residual_and_flow_nrmse_and_cosine",
                    "fisher_weighted_residual",
                    "held_direction_jvp_relative_error",
                    "finite_displacement_error",
                    "chart_reconstruction_occupancy_entropy_and_dead_regions",
                    "family_reuse_fanout_and_fanin",
                    "zero_transport_and_shuffled_gains",
                    "first_divergence_invocation",
                ],
                "causality": [
                    "future_token_perturbation",
                    "variable_sequence_length",
                    "left_and_right_padding",
                    "batch_composition",
                    "deterministic_replay",
                ],
            },
            "resource_accounting": {
                "count_unique_learned_scalars": True,
                "count_fixed_coefficients_and_codebooks": True,
                "primary_stored_scalar_footprint": (
                    "unique_learned_plus_fixed_inference_scalars"
                ),
                "count_chart_bases_centers_metrics_and_routers": True,
                "count_all_transports_corrections_and_sparse_indices": True,
                "count_legitimately_retained_exact_components": True,
                "count_unaccounted_or_live_fallbacks": True,
                "deduplicate_only_proven_physical_aliases": True,
                "equal_values_without_alias_are_deduplicated": False,
                "tied_source_embedding_and_head_counted_once": True,
                "candidate_shared_lexical_factor_counted_once": True,
                "negative_savings_are_preserved_not_clamped": True,
                "whole_model_and_target_component_ratios_reported": True,
                "expected_and_worst_case_conditional_macs_reported": True,
                "causal_pair_workload_keyed_by_sequence_operator_contract": True,
                "mixed_global_and_sliding_attention_priced_separately": True,
                "workload_pair_counts_bound_to_sequence_mask_receipts": True,
                "deployment_artifact_component_spans_must_close_exactly": True,
                "serialized_bytes_come_from_audited_artifact_spans": True,
                "logical_macs_and_executed_kernel_costs_separate": True,
                "peak_live_carrier_and_activation_bytes_reported": True,
                "counted_exact_components_disqualify_partial_compression": False,
                "unaccounted_teacher_duplicate_disqualifies_compression": True,
                "reachable_native_fallback_disqualifies_compression": True,
                "generic_resource_eligibility_authorizes_gemma_compression": (
                    False
                ),
                "gemma_bridge_and_family_disjoint_guard_required": True,
                "latency_inferred_from_macs": False,
            },
            "acceptance_gates": {
                "standalone_exact_control": {
                    "top1_agreement_minimum": 1.0,
                    "absolute_delta_nll_per_token_maximum": 0.0,
                    "native_to_candidate_kl_per_token_maximum": 0.0,
                    "valid_query_logit_maximum_absolute_error": 0.0,
                    "zero_live_teacher_calls_required": True,
                    "copied_checkpoint_coefficients_are_counted": True,
                    "all_causal_audits_must_pass": True,
                },
                "full_rank_chart_identity_control": {
                    "top1_agreement_minimum": 1.0,
                    "absolute_delta_nll_per_token_maximum": 1.0e-7,
                    "native_to_candidate_kl_per_token_maximum": 1.0e-7,
                    "valid_query_logit_maximum_absolute_error": 1.0e-6,
                    "all_causal_audits_must_pass": True,
                },
                "candidate_fidelity": {
                    "top1_agreement_minimum": 0.90,
                    "delta_nll_per_token_maximum": 0.10,
                    "native_to_candidate_kl_per_token_maximum": 0.10,
                    "mean_downstream_accuracy_retained_minimum": 0.90,
                    "worst_family_bootstrap_lower_bound_minimum": 0.85,
                },
                "compression": {
                    "whole_model_unique_parameter_reduction_minimum": 0.20,
                    "serialized_byte_reduction_minimum": 0.20,
                    "logical_mac_reduction_for_compute_claim_minimum": 0.10,
                    "must_match_or_beat_svd_and_deletion_at_same_budget": True,
                    "real_transport_gain_must_be_positive": True,
                    "shuffled_gain_fraction_of_real_maximum": 0.25,
                },
                "latency_or_speed_claim_without_measurement": False,
            },
            "external_rate_references": {
                "native_fp16_storage": True,
                "deterministic_weight_only_int8": True,
                "deterministic_groupwise_int4": True,
                "logical_mac_reduction_credit": False,
                "graph_plus_quantization_hybrid_in_v1": False,
                "primary_graph_candidate_storage_encoding": "float32",
                "primary_graph_quantized_storage_allowed": False,
            },
            "known_limits": {
                "dense_carrier_control_materializes_width_640_each_mutation": True,
                "residual_carrier_swap_alone_compresses_generator_compute": False,
                "new_carrier_native_modal_mutation_interface_required": True,
                "rank_is_capacity_not_automatic_storage_reduction": True,
                "basis_and_transport_overhead_can_erase_savings": True,
                "tokenizer_is_external_to_model_compilation": True,
            },
            "claim_boundary": {
                "setup_only": True,
                "model_or_prompt_access_occurred": False,
                "fitting_or_evaluation_occurred": False,
                "compression_success_claimed": False,
                "lossless_compression_claimed": False,
                "latency_or_kernel_speed_claimed": False,
            },
        }
        safe = _json_safe(state)
        if not isinstance(safe, dict):
            raise AssertionError("protocol state must be a mapping")
        safe["protocol_sha256"] = _sha256(safe)
        return safe


def build_layer_agnostic_whole_model_compression_protocol() -> dict[str, object]:
    return LayerAgnosticWholeModelCompressionProtocol().state_dict()


def validate_layer_agnostic_whole_model_compression_protocol(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly authenticate the single preregistered v1 setup artifact."""

    if not isinstance(value, Mapping):
        raise TypeError("layer-agnostic protocol must be a mapping")
    observed = _json_safe(dict(value))
    expected = build_layer_agnostic_whole_model_compression_protocol()
    if observed != expected:
        raise ValueError(
            "layer-agnostic whole-model compression protocol does not match "
            "the frozen v1 preregistration"
        )
    if observed.get("protocol_sha256") != _sha256(
        {
            key: item
            for key, item in observed.items()
            if key != "protocol_sha256"
        }
    ):
        raise ValueError("layer-agnostic protocol SHA-256 is invalid")
    return dict(observed)


def load_layer_agnostic_whole_model_compression_protocol(
    source: Path,
) -> dict[str, object]:
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("layer-agnostic protocol is not valid JSON") from error
    return validate_layer_agnostic_whole_model_compression_protocol(value)


def write_layer_agnostic_whole_model_compression_protocol(
    destination: Path = DEFAULT_PROTOCOL_OUTPUT,
) -> dict[str, object]:
    payload = build_layer_agnostic_whole_model_compression_protocol()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing protocol: {destination}"
        ) from error
    return payload


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the setup-only Gemma layer-agnostic whole-model "
            "compression protocol without loading a model or prompts."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROTOCOL_OUTPUT,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate an existing protocol instead of writing it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        payload = load_layer_agnostic_whole_model_compression_protocol(
            args.output
        )
        action = "checked"
    else:
        payload = write_layer_agnostic_whole_model_compression_protocol(
            args.output
        )
        action = "written"
    print(
        json.dumps(
            {
                "action": action,
                "output": str(args.output),
                "protocol_sha256": payload["protocol_sha256"],
                "setup_only": True,
                "model_weights_loaded": False,
                "experiment_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_PROTOCOL_OUTPUT",
    "LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_FORMAT_VERSION",
    "LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_SCHEMA",
    "LayerAgnosticWholeModelCompressionProtocol",
    "PINNED_FINAL_RMSNORM_PARAMETER_COUNT",
    "PINNED_HIDDEN_WIDTH",
    "PINNED_LAYER_COUNT",
    "PINNED_MLP_WIDTH",
    "PINNED_MODEL_ID",
    "PINNED_MODEL_REVISION",
    "PINNED_SOURCE_UNIQUE_PARAMETER_COUNT",
    "PINNED_TIED_LEXICAL_PARAMETER_COUNT",
    "PINNED_TRANSFORMER_BODY_PARAMETER_COUNT",
    "PINNED_VOCABULARY_SIZE",
    "build_layer_agnostic_whole_model_compression_protocol",
    "load_layer_agnostic_whole_model_compression_protocol",
    "main",
    "validate_layer_agnostic_whole_model_compression_protocol",
    "write_layer_agnostic_whole_model_compression_protocol",
]


if __name__ == "__main__":
    raise SystemExit(main())
