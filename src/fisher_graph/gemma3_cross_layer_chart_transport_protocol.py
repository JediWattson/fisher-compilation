"""Frozen setup contract for progressive cross-layer chart transport.

The protocol is deliberately model-run neutral.  It preregisters the arms,
fit order, lineage, controls, resource accounting, and claim boundaries for a
future pinned Gemma execution without loading weights or opening evaluation
data during setup.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_cross_layer_chart_transport.protocol.v1"
)
CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_FORMAT_VERSION = 1
DEFAULT_PROTOCOL_OUTPUT = Path(
    "artifacts/research/gemma3-cross-layer-chart-transport-protocol-v1.json"
)
_PROTOCOL_HASH_DOMAIN = (
    b"fisher_graph:gemma3-cross-layer-chart-transport-protocol:v1\0"
)
_ROW_HASH_DOMAIN = b"fisher_graph:aligned-chart-transport-rows:v1\0"
_PREFIX_HASH_DOMAIN = b"fisher_graph:compiled-chart-prefix-catalog:v1\0"
_STAGE_HASH_DOMAIN = b"fisher_graph:compiled-chart-transport-stage:v1\0"

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_DEVELOPMENT_PANEL_PATH = Path(
    "examples/gemma3_downstream_retention_v1.json"
)
PINNED_DEVELOPMENT_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)


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


def aligned_transport_row_identity_sha256(
    row_keys: Sequence[tuple[str, int]],
) -> str:
    keys = tuple(row_keys)
    if not keys or any(
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(key[0], str)
        or not key[0]
        or type(key[1]) is not int
        or key[1] < 0
        for key in keys
    ):
        raise ValueError("transport row keys must be nonempty example/position pairs")
    if len(set(keys)) != len(keys):
        raise ValueError("transport row keys must be unique")
    return _sha256(_ROW_HASH_DOMAIN, keys)


def require_aligned_transport_rows(
    source_row_keys: Sequence[tuple[str, int]],
    target_row_keys: Sequence[tuple[str, int]],
) -> str:
    source = tuple(source_row_keys)
    target = tuple(target_row_keys)
    if source != target:
        raise ValueError("source and target transport rows are not exactly aligned")
    return aligned_transport_row_identity_sha256(source)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def compiled_chart_transport_stage_sha256(
    *,
    layer_ordinal: int,
    layer_artifact_sha256: str,
    incoming_edge_artifact_sha256s: Sequence[str],
) -> str:
    edge_artifacts = tuple(incoming_edge_artifact_sha256s)
    if (
        type(layer_ordinal) is not int
        or layer_ordinal < 0
        or not _is_sha256(layer_artifact_sha256)
        or edge_artifacts != tuple(sorted(set(edge_artifacts)))
        or any(not _is_sha256(value) for value in edge_artifacts)
        or (layer_ordinal == 0 and edge_artifacts)
    ):
        raise ValueError("compiled chart transport stage lineage is invalid")
    return _sha256(
        _STAGE_HASH_DOMAIN,
        {
            "layer_ordinal": layer_ordinal,
            "layer_artifact_sha256": layer_artifact_sha256,
            "incoming_edge_artifact_sha256s": edge_artifacts,
        },
    )


def compiled_chart_prefix_catalog_sha256(
    stage_artifact_sha256s: Sequence[str],
) -> str:
    stages = tuple(stage_artifact_sha256s)
    if any(not _is_sha256(value) for value in stages):
        raise ValueError("compiled chart prefix lineage is invalid")
    return _sha256(
        _PREFIX_HASH_DOMAIN,
        {"stage_artifact_sha256s": stages},
    )


@dataclass(frozen=True, slots=True)
class CrossLayerChartTransportProtocol:
    layer_count: int = 18
    hidden_width: int = 640
    mode_count: int = 2048
    chart_count: int = 8
    chart_coordinate_rank: int = 16
    generator_rank_ladder: tuple[int, ...] = (2, 4, 8, 16)
    chart_temperature_ladder: tuple[float, ...] = (0.5, 1.0, 2.0)
    transport_ridge_ladder: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    maximum_incoming_charts_per_target_ladder: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        for label, value in (
            ("layer_count", self.layer_count),
            ("hidden_width", self.hidden_width),
            ("mode_count", self.mode_count),
            ("chart_count", self.chart_count),
            ("chart_coordinate_rank", self.chart_coordinate_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        ranks = self.generator_rank_ladder
        if (
            not ranks
            or ranks != tuple(sorted(set(ranks)))
            or any(type(value) is not int or value <= 0 for value in ranks)
            or ranks[-1] > self.mode_count
        ):
            raise ValueError("generator rank ladder is invalid")
        ridges = self.transport_ridge_ladder
        if (
            not ridges
            or ridges != tuple(sorted(set(ridges)))
            or any(
                not isinstance(value, float)
                or not math.isfinite(value)
                or value <= 0.0
                for value in ridges
            )
        ):
            raise ValueError("transport ridge ladder is invalid")
        temperatures = self.chart_temperature_ladder
        if (
            not temperatures
            or temperatures != tuple(sorted(set(temperatures)))
            or any(
                not isinstance(value, float)
                or not math.isfinite(value)
                or value <= 0.0
                for value in temperatures
            )
        ):
            raise ValueError("chart temperature ladder is invalid")
        indegrees = self.maximum_incoming_charts_per_target_ladder
        if (
            not indegrees
            or indegrees != tuple(sorted(set(indegrees)))
            or any(
                type(value) is not int
                or value <= 0
                or value > self.chart_count
                for value in indegrees
            )
        ):
            raise ValueError("transport indegree ladder is invalid")

    @property
    def chart_feature_width(self) -> int:
        return self.chart_coordinate_rank + 1

    def local_factorized_coefficients_per_layer(
        self,
        generator_rank: int,
    ) -> int:
        if generator_rank not in self.generator_rank_ladder:
            raise ValueError("generator rank is outside the frozen ladder")
        return self.chart_count * generator_rank * (
            self.chart_feature_width + self.mode_count
        )

    def maximum_transport_coefficients_per_layer(
        self,
        generator_rank: int,
        maximum_incoming_charts_per_target: int,
    ) -> int:
        if generator_rank not in self.generator_rank_ladder:
            raise ValueError("generator rank is outside the frozen ladder")
        if (
            maximum_incoming_charts_per_target
            not in self.maximum_incoming_charts_per_target_ladder
        ):
            raise ValueError("transport indegree is outside the frozen ladder")
        return (
            self.chart_count
            * maximum_incoming_charts_per_target
            * generator_rank
            * generator_rank
        )

    def state_dict(self) -> dict[str, object]:
        arms = (
            {
                "arm_id": "native",
                "all_native_layers_executed": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "exact_all_mode_graph",
                "all_modes_retained": True,
                "source_free_runtime": True,
                "bit_exact_native_equivalence_required": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "one_shot_independent_dense_chart_reproduction",
                "fit_trajectory": "all_native",
                "cross_layer_messages": False,
                "purpose": "reproduce_full_model_v1_failure",
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "progressive_independent_factorized_chart",
                "fit_trajectory": "actual_frozen_compiled_prefix",
                "cross_layer_messages": False,
                "purpose": "matched_progressive_baseline",
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "progressive_factorized_chart_transport",
                "fit_trajectory": "actual_frozen_compiled_prefix",
                "cross_layer_messages": True,
                "transport_scope": "adjacent_layer_same_token_only",
                "local_capacity_unchanged": True,
                "purpose": "dependency_and_fidelity_diagnostic",
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "budget_matched_factorized_chart_transport",
                "fit_trajectory": "actual_frozen_compiled_prefix",
                "cross_layer_messages": True,
                "local_generator_rank_reduced": True,
                "stored_scalars_and_macs_must_not_exceed": (
                    "progressive_independent_factorized_chart"
                ),
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "transport_zero_edge_control",
                "nested_candidate": "progressive_factorized_chart_transport",
                "transport_weights_zeroed_without_refit": True,
                "compression_credit_allowed": False,
            },
            {
                "arm_id": "transport_row_shuffled_negative_control",
                "fit_only_source_packets_permuted_by_example": True,
                "evaluation_rows_never_permuted": True,
                "compression_credit_allowed": False,
            },
        )
        state: dict[str, object] = {
            "schema": CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_SCHEMA,
            "format_version": (
                CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_FORMAT_VERSION
            ),
            "artifact_role": "preregistered_setup_only",
            "model_weights_loaded": False,
            "model_experiment_executed": False,
            "layer_count": self.layer_count,
            "pinned_runtime": {
                "model_id": PINNED_MODEL_ID,
                "model_revision": PINNED_MODEL_REVISION,
                "tokenizer_revision": PINNED_MODEL_REVISION,
                "device": "cpu",
                "dtype": "float32",
                "deterministic_seed": 1729,
                "development_panel_path": str(
                    PINNED_DEVELOPMENT_PANEL_PATH
                ),
                "development_panel_sha256": (
                    PINNED_DEVELOPMENT_PANEL_SHA256
                ),
            },
            "hidden_width": self.hidden_width,
            "mode_count": self.mode_count,
            "chart_count": self.chart_count,
            "chart_coordinate_rank": self.chart_coordinate_rank,
            "chart_feature_width": self.chart_feature_width,
            "generator_rank_ladder": self.generator_rank_ladder,
            "chart_temperature_ladder": self.chart_temperature_ladder,
            "transport_ridge_ladder": self.transport_ridge_ladder,
            "maximum_incoming_charts_per_target_ladder": (
                self.maximum_incoming_charts_per_target_ladder
            ),
            "arms": arms,
            "generator_message_definition": {
                "local_chart_feature": "q_lc_times_[1,z_lc]",
                "local_latent": "[1,z_lc]_times_L_lc",
                "emitted_latent": (
                    "q_lc_times_(local_latent_plus_incoming_latent)"
                ),
                "transport": "emitted_lminus1_c_times_T_lminus1c_to_ld",
                "mode_residual": "sum_d_emitted_latent_ld_times_D_ld",
                "strictly_forward": True,
                "same_token_only": True,
                "chart_indices_have_cross_layer_identity": False,
            },
            "progressive_fit_order": {
                "layer_order": tuple(range(self.layer_count)),
                "layer_zero_has_incoming_transport": False,
                "capture_rule": (
                    "freeze_prefix_0_through_lminus1_then_capture_native_"
                    "target_l_on_actual_compiled_prefix"
                ),
                "freeze_before_advance": True,
                "all_native_capture_reuse_for_progressive_fit": False,
                "prefix_catalog_sha256_required_per_layer": True,
                "source_target_row_identity_sha256_must_match": True,
            },
            "data_firewall": {
                "development_panel_is_family_disjoint": False,
                "development_result_may_claim_heldout_confirmation": False,
                "future_fit_selection_guard_families_must_be_disjoint": True,
                "fit_may_build_charts_factors_and_edge_shortlist": True,
                "selection_may_choose_rank_indegree_ridge_temperature": True,
                "guard_may_refit_or_select": False,
                "guard_opened_only_after_full_stack_freeze": True,
                "assessment_or_test_opened_during_development": False,
            },
            "edge_selection": {
                "raw_cross_layer_correlation_is_sufficient": False,
                "criterion": (
                    "partial_selection_residual_gain_beyond_local_generator"
                ),
                "causal_confirmation": (
                    "held_direction_jvp_then_finite_displacement"
                ),
                "maximum_source_layer_gap": 1,
                "shortlist_tie_break": (
                    "descending_partial_gain_then_source_chart_index"
                ),
                "skip_layer_and_cross_token_edges_deferred": True,
            },
            "frozen_numerics": {
                "global_generator_rank_per_candidate_stack": True,
                "weighted_affine_solver": "dual_ridge_float64",
                "affine_relative_ridge": 1.0e-3,
                "local_chart_solver": "joint_weighted_ridge_float64",
                "local_chart_relative_ridge": 1.0e-2,
                "chart_geometry_kmeans_iterations": 12,
                "factorization": "per_chart_canonical_truncated_svd",
                "transport_solver": "deterministic_block_ridge",
                "block_coordinate_sweeps": 8,
                "monotonic_objective_tolerance": (
                    "128_times_float64_epsilon_times_max_1_baseline_sse"
                ),
                "row_fisher_clamp": (0.1, 10.0),
            },
            "guard_acceptance_thresholds": {
                "top1_agreement_to_native_minimum": 0.90,
                "delta_nll_per_token_maximum": 0.10,
                "native_to_candidate_kl_per_token_maximum": 0.10,
                "transport_gain_over_zero_edge_must_be_positive": True,
                "row_shuffled_gain_fraction_of_real_maximum": 0.25,
                "family_bootstrap_lower_bound_must_pass": True,
            },
            "required_metrics": (
                "modal_state_nrmse",
                "mlp_output_nrmse_and_cosine",
                "fisher_weighted_residual",
                "transport_partial_explained_fraction",
                "zero_edge_ablation_gain",
                "row_shuffled_negative_control_gain",
                "chart_membership_entropy_and_occupancy",
                "prefix_delta_nll_and_kl_after_each_layer",
                "full_model_nll_kl_top1_and_logit_nrmse",
                "worst_family_metrics_and_bootstrap_intervals",
                "held_direction_jvp_and_finite_displacement",
            ),
            "resource_accounting": {
                "count_affine_base_and_final_mode_decoder": True,
                "count_chart_centers_bases_scales": True,
                "count_local_L_and_D_factors": True,
                "count_sparse_transport_T_edges_and_indices": True,
                "count_chart_routing_and_transport_macs": True,
                "count_peak_live_packet_bytes": True,
                "softmax_non_mac_ops_reported_separately": True,
                "theoretical_sparse_and_realized_kernel_costs_separate": True,
                "source_weight_removal_credit_requires_source_free_artifact": True,
            },
            "claim_boundary": {
                "all_modes_retained": True,
                "deletion_performed": False,
                "merging_performed": False,
                "dependency_proven_by_prediction_alone": False,
                "compression_success_claimed": False,
                "latency_or_kernel_speed_claimed": False,
                "transport_plus_unchanged_local_is_accuracy_only": True,
                "compression_requires_budget_matched_guard_pass": True,
            },
            "known_capacity_limit": {
                "affine_base_and_mode_decoder_remain_dense": True,
                "transport_can_only_compact_chart_correction_in_v1": True,
                "large_future_gain_requires_base_or_decoder_sharing": True,
            },
        }
        state["protocol_sha256"] = _sha256(_PROTOCOL_HASH_DOMAIN, state)
        return state


def build_cross_layer_chart_transport_protocol() -> dict[str, object]:
    return CrossLayerChartTransportProtocol().state_dict()


__all__ = [
    "CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_FORMAT_VERSION",
    "CROSS_LAYER_CHART_TRANSPORT_PROTOCOL_SCHEMA",
    "CrossLayerChartTransportProtocol",
    "DEFAULT_PROTOCOL_OUTPUT",
    "aligned_transport_row_identity_sha256",
    "build_cross_layer_chart_transport_protocol",
    "compiled_chart_prefix_catalog_sha256",
    "compiled_chart_transport_stage_sha256",
    "PINNED_DEVELOPMENT_PANEL_PATH",
    "PINNED_DEVELOPMENT_PANEL_SHA256",
    "PINNED_MODEL_ID",
    "PINNED_MODEL_REVISION",
    "require_aligned_transport_rows",
]
