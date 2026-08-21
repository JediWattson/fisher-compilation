"""Preregister the exact-model edge-sidecar causality rung.

This module is deliberately standard-library only.  It freezes a future
Gemma experiment in which the complete 39-event model remains exact and every
chart and edge is observational.  Sidecar predictions may be measured against
exact activations, JVPs, and isolated shadow interventions, but they may never
be applied to the baseline carrier or logits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


EXACT_EDGE_SIDECAR_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_exact_edge_sidecar.protocol.v1"
)
EXACT_EDGE_SIDECAR_PROTOCOL_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-exact-edge-sidecar-protocol-v1.json"
)
_HASH_DOMAIN = b"fisher_graph:gemma3-exact-edge-sidecar-protocol:v1\0"

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_PARAMETER_COUNT = 268_098_176
PINNED_HIDDEN_WIDTH = 640
PINNED_LAYER_COUNT = 18
PINNED_EVENT_COUNT = 39
PINNED_CHART_SITE_COUNT = 73
PINNED_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_SPLIT_ID_SHA256S = {
    "A_fit": "29de4c8600275ce2ce597dd4733e7ce59731c4caedf5102d425489339067ec55",
    "B_predictive": (
        "d27459dc74caae545d73bef8141cac35065ed5a9806715df66635bc296147352"
    ),
    "C_intervention": (
        "d09ec93e201de453107fb85bf8d9a0421e58641a7063cd3067c825fe77042ab2"
    ),
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _protocol_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _runtime_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event_id": "lexical.ingress",
            "event_index": 0,
            "event_kind": "exact_tied_lexical_ingress",
            "block_ordinal": None,
            "component_family": "lexical",
        }
    ]
    for block in range(PINNED_LAYER_COUNT):
        attention_index = 1 + 2 * block
        mlp_index = attention_index + 1
        events.extend(
            (
                {
                    "event_id": f"mutation.{attention_index}",
                    "event_index": attention_index,
                    "event_kind": "exact_attention_mutation",
                    "block_ordinal": block,
                    "component_family": "attention",
                    "attention_kind": (
                        "global_causal"
                        if block in (5, 11, 17)
                        else "sliding_causal_512"
                    ),
                },
                {
                    "event_id": f"mutation.{mlp_index}",
                    "event_index": mlp_index,
                    "event_kind": "exact_mlp_mutation",
                    "block_ordinal": block,
                    "component_family": "mlp",
                },
            )
        )
    events.extend(
        (
            {
                "event_id": "terminal.rmsnorm",
                "event_index": 37,
                "event_kind": "exact_terminal_rmsnorm",
                "block_ordinal": None,
                "component_family": "terminal",
            },
            {
                "event_id": "lexical.egress",
                "event_index": 38,
                "event_kind": "exact_tied_lexical_egress",
                "block_ordinal": None,
                "component_family": "lexical",
            },
        )
    )
    return events


def _chart_sites() -> list[dict[str, object]]:
    sites: list[dict[str, object]] = [
        {
            "site_id": "lexical.ingress.output",
            "site_index": 0,
            "event_id": "lexical.ingress",
            "event_index": 0,
            "block_ordinal": -1,
            "component_family": "lexical",
            "boundary_role": "output",
            "capture_tensor": "exact_lexical_ingress_carrier",
            "local_predecessor_site_id": None,
            "source_eligible": True,
            "target_eligible": False,
        }
    ]
    for block in range(PINNED_LAYER_COUNT):
        attention_event = f"mutation.{1 + 2 * block}"
        mlp_event = f"mutation.{2 + 2 * block}"
        definitions = (
            (
                f"{attention_event}.input",
                attention_event,
                "attention",
                "input",
                "exact_normalized_attention_input",
            ),
            (
                f"{attention_event}.output",
                attention_event,
                "attention",
                "output",
                "exact_post_output_norm_attention_residual_delta",
            ),
            (
                f"{mlp_event}.input",
                mlp_event,
                "mlp",
                "input",
                "exact_normalized_feed_forward_input",
            ),
            (
                f"{mlp_event}.output",
                mlp_event,
                "mlp",
                "output",
                "exact_post_output_norm_feed_forward_residual_delta",
            ),
        )
        for site_id, event_id, family, role, tensor_name in definitions:
            predecessor = sites[-1]["site_id"]
            sites.append(
                {
                    "site_id": site_id,
                    "site_index": len(sites),
                    "event_id": event_id,
                    "event_index": int(event_id.split(".")[1]),
                    "block_ordinal": block,
                    "component_family": family,
                    "boundary_role": role,
                    "capture_tensor": tensor_name,
                    "local_predecessor_site_id": predecessor,
                    "source_eligible": True,
                    "target_eligible": role == "output",
                }
            )
    return sites


def _arms() -> list[dict[str, object]]:
    shared = {
        "exact_model_forward": True,
        "model_weights_mutated": False,
        "model_carrier_mutated_by_sidecar": False,
        "model_logits_mutated_by_sidecar": False,
        "compression": False,
        "candidate_model_produced": False,
    }
    return [
        {
            **shared,
            "arm_id": "local_target_baseline",
            "edge_kind": "none",
            "edge_budget_matched_to_real_arm": False,
            "predictor": (
                "chart_conditioned_affine_from_immediate_exact_predecessor"
            ),
            "purpose": "measure_predictability_without_skip_edges",
        },
        {
            **shared,
            "arm_id": "real_forward_cross_block_edges",
            "edge_kind": "fitted_real",
            "edge_budget_matched_to_real_arm": True,
            "strictly_forward": True,
            "cross_block": True,
            "cross_component_family": True,
            "same_logical_token": True,
            "source_provenance_used_as_numerical_feature": False,
            "purpose": "primary_predictive_and_causal_edge_hypothesis",
        },
        {
            **shared,
            "arm_id": "zero_edge_control",
            "edge_kind": "real_topology_weights_hard_zero",
            "edge_budget_matched_to_real_arm": True,
            "topology_copied_from": "real_forward_cross_block_edges",
            "edge_weights_refitted": False,
            "purpose": "isolate_edge_application_from_local_baseline",
        },
        {
            **shared,
            "arm_id": "row_shuffled_control",
            "edge_kind": "fit_rows_deterministically_permuted",
            "edge_budget_matched_to_real_arm": True,
            "A_fit_source_rows_shuffled": True,
            "B_and_C_rows_shuffled": False,
            "purpose": "destroy_same_token_alignment",
        },
        {
            **shared,
            "arm_id": "family_shuffled_control",
            "edge_kind": "fit_prompt_families_deterministically_deranged",
            "edge_budget_matched_to_real_arm": True,
            "logical_position_distribution_matched": True,
            "B_and_C_rows_shuffled": False,
            "purpose": "destroy_semantic_family_alignment",
        },
        {
            **shared,
            "arm_id": "source_provenance_permutation_control",
            "edge_kind": "eligible_source_site_identity_derangement",
            "edge_budget_matched_to_real_arm": True,
            "source_tensor_multiset_unchanged": True,
            "source_site_labels_fixed": True,
            "packet_assignment_to_source_labels_deranged": True,
            "permutation_within_target_eligible_shape_bucket": True,
            "same_permutation_frozen_for_B_and_C": True,
            "purpose": (
                "test_whether_predictive_packet_is_causally_localized_to_"
                "the_claimed_source_site"
            ),
        },
    ]


@dataclass(frozen=True, slots=True)
class ExactEdgeSidecarProtocol:
    """Frozen setup contract; construction performs no experiment work."""

    chart_count_per_site: int = 1
    coordinate_rank: int = PINNED_HIDDEN_WIDTH
    edge_coefficient_rank: int = PINNED_HIDDEN_WIDTH
    maximum_real_incoming_edges_per_target: int = 1
    deterministic_seed: int = 1729
    ridge: float = 1.0e-4
    jvp_direction_count_per_edge: int = 8
    intervention_epsilon_ladder: tuple[float, ...] = (0.001, 0.003, 0.01)

    def __post_init__(self) -> None:
        for name in (
            "chart_count_per_site",
            "coordinate_rank",
            "edge_coefficient_rank",
            "maximum_real_incoming_edges_per_target",
            "deterministic_seed",
            "jvp_direction_count_per_edge",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.coordinate_rank != PINNED_HIDDEN_WIDTH:
            raise ValueError("edge-only v1 requires exact full-width coordinates")
        if self.edge_coefficient_rank != PINNED_HIDDEN_WIDTH:
            raise ValueError("edge-only v1 requires full-width edge coefficients")
        if self.chart_count_per_site != 1:
            raise ValueError("edge-only v1 freezes one full-width chart per site")
        if self.maximum_real_incoming_edges_per_target != 1:
            raise ValueError("edge-only v1 freezes one incoming skip edge per target")
        if self.deterministic_seed != 1729:
            raise ValueError("edge-only v1 deterministic seed is frozen")
        if self.jvp_direction_count_per_edge != 8:
            raise ValueError("edge-only v1 JVP direction count is frozen")
        if type(self.ridge) is not float or self.ridge != 1.0e-4:
            raise ValueError("edge-only v1 ridge is frozen")
        if self.intervention_epsilon_ladder != (0.001, 0.003, 0.01):
            raise ValueError("edge-only v1 intervention ladder is frozen")

    def state_dict(self) -> dict[str, object]:
        events = _runtime_events()
        sites = _chart_sites()
        arms = _arms()
        state: dict[str, object] = {
            "schema": EXACT_EDGE_SIDECAR_PROTOCOL_SCHEMA,
            "format_version": EXACT_EDGE_SIDECAR_PROTOCOL_FORMAT_VERSION,
            "artifact_role": "setup_only_preregistered_edge_axis",
            "setup_receipt": {
                "model_weights_loaded": False,
                "prompt_data_opened": False,
                "tokenization_performed": False,
                "activations_captured": False,
                "charts_fitted": False,
                "edges_fitted": False,
                "JVPs_computed": False,
                "finite_interventions_executed": False,
                "experiment_executed": False,
                "metrics_present": False,
            },
            "scope_firewall": {
                "axis": "edge_structure_only",
                "compression": False,
                "compression_axis_enabled": False,
                "compressed_candidate_present": False,
                "parameter_removal": False,
                "parameter_merging": False,
                "rank_reduction": False,
                "weight_approximation": False,
                "all_model_weights_exact": True,
                "all_model_ranks_exact": True,
                "reference_model_event_count": PINNED_EVENT_COUNT,
                "baseline_model_output_must_remain_byte_identical": True,
                "sidecar_edges_may_write_model_state": False,
                "sidecar_edges_may_write_model_logits": False,
                "combined_edge_and_compression_arm_allowed": False,
            },
            "pinned_runtime": {
                "model_id": PINNED_MODEL_ID,
                "model_revision": PINNED_MODEL_REVISION,
                "tokenizer_revision": PINNED_MODEL_REVISION,
                "source_unique_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
                "layer_count": PINNED_LAYER_COUNT,
                "hidden_width": PINNED_HIDDEN_WIDTH,
                "device": "cpu",
                "dtype": "float32",
                "attention_implementation": "eager",
                "deterministic_seed": self.deterministic_seed,
            },
            "exact_39_event_gate": {
                "events": events,
                "event_ids": [event["event_id"] for event in events],
                "event_count": len(events),
                "attention_mutation_count": 18,
                "mlp_mutation_count": 18,
                "global_attention_block_ordinals": [5, 11, 17],
                "tied_lexical_ingress_and_egress_exact": True,
                "residual_carrier_initial_version": 0,
                "residual_carrier_final_version": 36,
                "source_parameter_inventory_parity_required": True,
                "sidecar_disabled_vs_enabled": {
                    "event_order_equal": True,
                    "every_mutation_delta_byte_identical": True,
                    "final_carrier_byte_identical": True,
                    "normalized_output_byte_identical": True,
                    "logits_byte_identical": True,
                    "delta_nll_per_token": 0.0,
                    "native_to_sidecar_kl_per_query": 0.0,
                    "top1_agreement": 1.0,
                    "source_state_fingerprint_unchanged": True,
                },
            },
            "unified_exact_chart_contract": {
                "site_count": len(sites),
                "source_eligible_site_count": sum(
                    bool(site["source_eligible"]) for site in sites
                ),
                "target_eligible_site_count": sum(
                    bool(site["target_eligible"]) for site in sites
                ),
                "sites": sites,
                "site_count_by_component_and_role": {
                    "lexical_output": 1,
                    "attention_input": 18,
                    "attention_output": 18,
                    "mlp_input": 18,
                    "mlp_output": 18,
                },
                "site_order_is_strict": True,
                "chart_count_per_site": self.chart_count_per_site,
                "hidden_width": PINNED_HIDDEN_WIDTH,
                "coordinate_rank": self.coordinate_rank,
                "coordinate_rank_is_full_width": True,
                "identity_basis_may_be_implicit_not_stored": True,
                "chart_projection_is_rank_reduced": False,
                "chart_coordinate_reconstruction_is_exact": True,
                "chart_fitting_changes_model_execution": False,
                "capture_shape": "batch_by_sequence_by_640",
                "capture_rows_require_example_id_and_logical_position": True,
                "capture_callback_is_observational_and_detached": True,
                "callback_return_value_is_ignored": True,
            },
            "predictive_task": {
                "target": (
                    "exact_attention_or_MLP_output_residual_delta_"
                    "full_width_coordinates"
                ),
                "target_eligible_site_count": 36,
                "input_sites_are_sources_but_not_prediction_targets": True,
                "lexical_ingress_is_source_but_not_prediction_target": True,
                "local_baseline": (
                    "chart_conditioned_affine_from_paired_same_event_"
                    "exact_input_site"
                ),
                "real_edge_addition": (
                    "one_strictly_earlier_cross_block_same_token_source"
                ),
                "source_block_must_precede_target_block": True,
                "source_component_family_must_differ_from_target": True,
                "lexical_ingress_source_ordinal": -1,
                "layer_distance_is_a_numerical_feature": False,
                "source_event_id_is_a_numerical_feature": False,
                "source_component_family_is_a_numerical_feature": False,
                "provenance_may_only_filter_edge_eligibility": True,
                "source_label_may_break_numerical_score_ties": False,
                "score_tie_break": (
                    "canonical_hash_of_design_tensor_bytes_then_chart_index"
                ),
                "label_only_renaming_invariance_required": True,
                "fit_objective": "Fisher_weighted_ridge_SSE_on_A_fit",
                "edge_coefficient_rank": self.edge_coefficient_rank,
                "edge_coefficient_shape": "640_by_640",
                "edge_low_rank_factorization_used": False,
                "edge_numerical_rank_may_be_design_limited": True,
                "edge_numerical_rank_must_be_reported": True,
                "edge_selection_role": "A_fit_only",
                "maximum_incoming_edges_per_target": (
                    self.maximum_real_incoming_edges_per_target
                ),
                "relative_ridge": self.ridge,
                "all_edge_bearing_controls_share_real_edge_budget": True,
                "every_arm_uses_the_same_local_baseline": True,
                "edges_are_analysis_sidecars_only": True,
                "edges_are_applied_to_exact_model_forward": False,
            },
            "arms": arms,
            "arm_order": [arm["arm_id"] for arm in arms],
            "data_firewall": {
                "panel_path": str(PINNED_PANEL_PATH),
                "panel_sha256": PINNED_PANEL_SHA256,
                "family_count": 6,
                "total_example_count": 60,
                "example_id_hash_encoding": (
                    "sha256_of_canonical_JSON_UTF8_ordered_ID_array"
                ),
                "split_algorithm": (
                    "sort_families_then_preserve_panel_order_within_family;"
                    "examples_0_to_5_A_fit;6_to_7_B_predictive;"
                    "8_to_9_C_intervention"
                ),
                "roles": {
                    "A_fit": {
                        "example_count": 36,
                        "example_ids_sha256": PINNED_SPLIT_ID_SHA256S["A_fit"],
                        "may_fit_charts_local_baselines_edges_and_controls": True,
                    },
                    "B_predictive": {
                        "example_count": 12,
                        "example_ids_sha256": (
                            PINNED_SPLIT_ID_SHA256S["B_predictive"]
                        ),
                        "may_select_or_refit": False,
                        "purpose": "heldout_predictive_gain_gate",
                    },
                    "C_intervention": {
                        "example_count": 12,
                        "example_ids_sha256": (
                            PINNED_SPLIT_ID_SHA256S["C_intervention"]
                        ),
                        "may_select_or_refit": False,
                        "purpose": "heldout_JVP_and_finite_intervention_gate",
                    },
                },
                "example_and_prompt_disjoint_across_roles": True,
                "families_overlap_across_roles": True,
                "family_disjoint_confirmation": False,
                "B_and_C_rows_may_enter_fit": False,
                "B_metrics_may_change_edge_selection": False,
                "C_metrics_may_change_edge_selection": False,
                "raw_prompt_text_or_token_ids_written_to_report": False,
                "row_identity_sha256_required_for_every_capture": True,
            },
            "heldout_predictive_gate": {
                "role": "B_predictive",
                "primary_metric": "Fisher_weighted_relative_SSE_gain_vs_local",
                "minimum_real_gain_vs_local": 0.02,
                "minimum_real_gain_margin_over_each_negative_control": 0.01,
                "paired_family_bootstrap_lower_bound_must_exceed_zero": True,
                "bootstrap_seed": self.deterministic_seed,
                "bootstrap_resamples": 10_000,
                "edge_catalog_is_not_pruned_or_refitted_on_B": True,
                "any_selected_edge_with_nonpositive_B_gain_fails_the_arm": True,
            },
            "heldout_causal_gate": {
                "role": "C_intervention",
                "JVP": {
                    "direction_count_per_edge": self.jvp_direction_count_per_edge,
                    "directions": (
                        "deterministic_Rademacher_minus_one_plus_one_on_valid_"
                        "source_rows_with_RMS_exactly_one_not_selected_on_C"
                    ),
                    "invalid_source_rows": "exact_zero",
                    "finite_displacement_scale": (
                        "epsilon_times_baseline_source_site_RMS_times_"
                        "Rademacher_direction"
                    ),
                    "exact_autograd_reference_required": True,
                    "minimum_real_direction_cosine": 0.90,
                    "maximum_real_relative_norm_error": 0.25,
                    "minimum_cosine_margin_over_each_negative_control": 0.05,
                },
                "finite_intervention": {
                    "shadow_execution_only": True,
                    "baseline_model_output_is_never_replaced": True,
                    "source_activation_perturbation_is_ephemeral": True,
                    "source_model_weights_remain_exact_and_frozen": True,
                    "source_state_fingerprint_restored_after_each_probe": True,
                    "intervened_logits_are_diagnostic_not_candidate_output": True,
                    "central_plus_minus_epsilon": True,
                    "epsilon_times_source_RMS": list(
                        self.intervention_epsilon_ladder
                    ),
                    "minimum_real_direction_cosine": 0.80,
                    "maximum_real_relative_norm_error": 0.35,
                    "minimum_cosine_margin_over_each_negative_control": 0.05,
                    "maximum_adjacent_epsilon_derivative_disagreement": 0.10,
                },
                "predictive_gain_alone_authorizes_causal_language": False,
                "both_JVP_and_finite_intervention_must_pass": True,
            },
            "artifact_chain": {
                "hash_algorithm": "sha256_canonical_json_and_tensor_bytes",
                "every_artifact_requires_own_sha256": True,
                "every_downstream_artifact_requires_parent_sha256s": True,
                "fixed_artifacts": [
                    {
                        "artifact_id": "exact_chart_catalog",
                        "default_path": (
                            ".local-runs/google--gemma-3-270m/"
                            "exact-edge-sidecar/chart-catalog-v1.json"
                        ),
                        "schema": (
                            "fisher_graph.gemma3_exact_edge_sidecar."
                            "chart_catalog.v1"
                        ),
                        "contains_model_weights": False,
                        "contains_prompt_text_or_token_ids": False,
                    },
                    {
                        "artifact_id": "frozen_edge_arm_catalog",
                        "default_path": (
                            ".local-runs/google--gemma-3-270m/"
                            "exact-edge-sidecar/edge-arm-catalog-v1.json"
                        ),
                        "schema": (
                            "fisher_graph.gemma3_exact_edge_sidecar."
                            "edge_arm_catalog.v1"
                        ),
                        "parent_artifacts": ["exact_chart_catalog"],
                        "edge_coefficients_are_analysis_only": True,
                    },
                    {
                        "artifact_id": "heldout_predictive_receipt",
                        "default_path": (
                            ".local-runs/google--gemma-3-270m/"
                            "exact-edge-sidecar/predictive-receipt-v1.json"
                        ),
                        "schema": (
                            "fisher_graph.gemma3_exact_edge_sidecar."
                            "predictive_receipt.v1"
                        ),
                        "parent_artifacts": ["frozen_edge_arm_catalog"],
                    },
                    {
                        "artifact_id": "heldout_causal_receipt",
                        "default_path": (
                            ".local-runs/google--gemma-3-270m/"
                            "exact-edge-sidecar/causal-receipt-v1.json"
                        ),
                        "schema": (
                            "fisher_graph.gemma3_exact_edge_sidecar."
                            "causal_receipt.v1"
                        ),
                        "parent_artifacts": [
                            "frozen_edge_arm_catalog",
                            "heldout_predictive_receipt",
                        ],
                    },
                    {
                        "artifact_id": "edge_axis_report",
                        "default_path": (
                            ".local-runs/google--gemma-3-270m/"
                            "exact-edge-sidecar/report-v1.json"
                        ),
                        "schema": (
                            "fisher_graph.gemma3_exact_edge_sidecar.report.v1"
                        ),
                        "parent_artifacts": [
                            "exact_chart_catalog",
                            "frozen_edge_arm_catalog",
                            "heldout_predictive_receipt",
                            "heldout_causal_receipt",
                        ],
                    },
                ],
            },
            "analysis_cost_and_claim_firewall": {
                "source_model_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
                "model_parameter_reduction_fraction": 0.0,
                "model_storage_reduction_fraction": 0.0,
                "model_MAC_reduction_fraction": 0.0,
                "compression": False,
                "compression_success_claimed": False,
                "latency_improvement_claimed": False,
                "sidecar_chart_scalar_count_must_be_reported": True,
                "sidecar_edge_scalar_count_must_be_reported": True,
                "fit_FLOPs_or_MACs_must_be_reported": True,
                "exact_forward_count_must_be_reported": True,
                "backward_or_JVP_count_must_be_reported": True,
                "finite_intervention_forward_count_must_be_reported": True,
                "peak_resident_bytes_must_be_reported": True,
                "sidecar_cost_may_not_be_subtracted_from_model_cost": True,
                "edge_predictive_success_is_not_compression_success": True,
            },
            "promotion_firewall": {
                "edge_axis_pass_requires_exact_gate": True,
                "edge_axis_pass_requires_predictive_gate": True,
                "edge_axis_pass_requires_JVP_gate": True,
                "edge_axis_pass_requires_finite_intervention_gate": True,
                "combined_edge_and_compression_arm_locked": True,
                "combined_arm_unlock_condition": (
                    "this_exact_edge_axis_passes_and_a_separately_"
                    "preregistered_compression_axis_passes"
                ),
                "this_setup_authorizes_model_mutation": False,
                "this_setup_authorizes_compression_claim": False,
            },
        }
        state["protocol_sha256"] = _protocol_sha256(state)
        return state


def build_exact_edge_sidecar_protocol() -> dict[str, object]:
    return ExactEdgeSidecarProtocol().state_dict()


def validate_exact_edge_sidecar_protocol(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("exact edge-sidecar protocol must be a mapping")
    observed = json.loads(_canonical_json_bytes(dict(value)))
    expected = build_exact_edge_sidecar_protocol()
    if observed != expected:
        raise ValueError("exact edge-sidecar protocol does not match frozen v1")
    digest = observed.get("protocol_sha256")
    without_digest = {
        key: item for key, item in observed.items() if key != "protocol_sha256"
    }
    if digest != _protocol_sha256(without_digest):
        raise ValueError("exact edge-sidecar protocol SHA-256 is invalid")
    return dict(observed)


def load_exact_edge_sidecar_protocol(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("exact edge-sidecar protocol is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("exact edge-sidecar protocol must be a JSON object")
    return validate_exact_edge_sidecar_protocol(value)


def write_exact_edge_sidecar_protocol(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(destination)
    payload = build_exact_edge_sidecar_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write or authenticate the setup-only exact Gemma edge-sidecar "
            "causality protocol."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate an existing frozen artifact instead of writing it",
    )
    args = parser.parse_args(argv)
    if args.check:
        payload = load_exact_edge_sidecar_protocol(args.output)
        action = "checked"
    else:
        payload = write_exact_edge_sidecar_protocol(args.output)
        action = "written"
    print(
        json.dumps(
            {
                "action": action,
                "output": str(args.output),
                "protocol_sha256": payload["protocol_sha256"],
                "setup_only": True,
                "compression": False,
                "model_weights_loaded": False,
                "experiment_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "EXACT_EDGE_SIDECAR_PROTOCOL_FORMAT_VERSION",
    "EXACT_EDGE_SIDECAR_PROTOCOL_SCHEMA",
    "ExactEdgeSidecarProtocol",
    "build_exact_edge_sidecar_protocol",
    "load_exact_edge_sidecar_protocol",
    "main",
    "validate_exact_edge_sidecar_protocol",
    "write_exact_edge_sidecar_protocol",
]


if __name__ == "__main__":
    raise SystemExit(main())
