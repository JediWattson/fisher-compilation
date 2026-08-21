"""Preregister the runnable Fisher/activation-chart whole-model worker.

This module is standard-library only.  It freezes the bounded first arm and
the callback/artifact contracts consumed by the model-facing worker without
loading Gemma, opening prompt data, tokenizing, fitting tensors, or executing
the experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

FISHER_ACTIVATION_CHART_SETUP_SCHEMA = (
    "fisher_graph.gemma3_fisher_activation_chart_whole_model_experiment."
    "setup.v1"
)
FISHER_ACTIVATION_CHART_SETUP_FORMAT_VERSION = 1
FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID = (
    "all_18_mlp_progressive_fisher_activation_chart_transport_"
    "development_overlay"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-fisher-activation-chart-whole-model-experiment-setup-v1.json"
)
_HASH_DOMAIN = b"fisher_graph:fisher-activation-chart-setup:v1\0"
_PINNED_LAYER_AGNOSTIC_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_layer_agnostic_whole_model_compression.protocol.v1"
)
_PINNED_LAYER_AGNOSTIC_PROTOCOL_SHA256 = (
    "4825f3e4e21475c54dae917d16a647c105f2d45afc191dec0b5689d0064d6107"
)
_PINNED_TRANSPORT_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_cross_layer_chart_transport.protocol.v1"
)
_PINNED_TRANSPORT_PROTOCOL_SHA256 = (
    "b373f855076c73aceb9fea9adf404a60cf7406718e4b7a3c96dc295b8a64ab5b"
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
    return hashlib.sha256(_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _runtime_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event_id": "lexical.ingress",
            "event_index": 0,
            "kind": "exact_tied_lexical_ingress",
            "token_dependency": "token_local_lookup",
        }
    ]
    for layer in range(18):
        for stage in ("attention", "feed_forward"):
            index = len(events)
            events.append(
                {
                    "event_id": f"mutation.{index}",
                    "event_index": index,
                    "kind": (
                        "exact_causal_attention"
                        if stage == "attention"
                        else "fisher_activation_chart_generator"
                    ),
                    "provenance_only": {
                        "source_layer_ordinal": layer,
                        "source_stage": stage,
                        "source_mutation_id": f"layer.{layer}.{stage}",
                    },
                    "token_dependency": (
                        "causal_prefix"
                        if stage == "attention"
                        else "token_local"
                    ),
                    "attention_contract": (
                        (
                            "global_theta_1000000"
                            if layer in (5, 11, 17)
                            else "sliding_512_theta_10000"
                        )
                        if stage == "attention"
                        else None
                    ),
                }
            )
    events.extend(
        (
            {
                "event_id": "terminal.rmsnorm",
                "event_index": 37,
                "kind": "exact_terminal_rmsnorm",
                "token_dependency": "token_local",
            },
            {
                "event_id": "lexical.egress",
                "event_index": 38,
                "kind": "exact_tied_lexical_egress",
                "token_dependency": "token_local_vocabulary_projection",
            },
        )
    )
    return events


@dataclass(frozen=True, slots=True)
class FisherActivationChartWholeModelExperimentSetup:
    """Frozen bounded arm and fail-closed worker orchestration contract."""

    chart_count: int = 8
    chart_rank: int = 16
    generator_rank: int = 4
    transport_rank: int = 4
    maximum_incoming_charts_per_target: int = 2
    batch_size: int = 4
    affine_ridge: float = 1.0e-3
    chart_ridge: float = 1.0e-2
    transport_ridge: float = 1.0e-4
    chart_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "chart_count",
            "chart_rank",
            "generator_rank",
            "transport_rank",
            "maximum_incoming_charts_per_target",
            "batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.chart_rank > 640 or self.generator_rank > 2048:
            raise ValueError("bounded worker rank exceeds source width")
        if self.generator_rank > self.chart_rank + 1:
            raise ValueError("generator_rank exceeds chart feature width")
        if self.transport_rank != self.generator_rank:
            raise ValueError("current transport executor requires square rank")
        for name in ("affine_ridge", "chart_ridge", "transport_ridge"):
            value = getattr(self, name)
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.chart_temperature, float) or (
            self.chart_temperature <= 0
            or not math.isfinite(self.chart_temperature)
        ):
            raise ValueError("chart_temperature must be positive")

    def state_dict(self) -> dict[str, object]:
        events = _runtime_events()
        state: dict[str, object] = {
            "schema": FISHER_ACTIVATION_CHART_SETUP_SCHEMA,
            "format_version": FISHER_ACTIVATION_CHART_SETUP_FORMAT_VERSION,
            "artifact_role": (
                "setup_only_for_runnable_bounded_development_overlay"
            ),
            "setup_receipt": {
                "model_weights_loaded": False,
                "prompt_data_opened": False,
                "tokenization_performed": False,
                "activation_capture_performed": False,
                "fisher_computed": False,
                "charts_fitted": False,
                "candidate_trajectory_recaptured": False,
                "transports_fitted": False,
                "experiment_executed": False,
                "metrics_present": False,
                "compression_success_claimed": False,
            },
            "protocol_binding": {
                "aspirational_layer_agnostic_protocol": {
                    "schema": _PINNED_LAYER_AGNOSTIC_PROTOCOL_SCHEMA,
                    "protocol_sha256": (
                        _PINNED_LAYER_AGNOSTIC_PROTOCOL_SHA256
                    ),
                    "implemented_by_this_worker": False,
                },
                "delegated_executable_transport_protocol": {
                    "schema": _PINNED_TRANSPORT_PROTOCOL_SCHEMA,
                    "protocol_sha256": _PINNED_TRANSPORT_PROTOCOL_SHA256,
                    "implemented_by_this_worker": True,
                },
                "implemented_bounded_arm": FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID,
            },
            "bounded_scope": {
                "development_overlay_only": True,
                "uses_native_tied_lexical_ingress_and_egress": True,
                "uses_native_attention_residual_and_normalization_path": True,
                "all_18_feed_forward_mutations_replaced": True,
                "replacement_kind": (
                    "progressive_fisher_weighted_factorized_activation_chart_"
                    "generators_with_adjacent_same_token_transports"
                ),
                "reference_transformer_event_count": 39,
                "replaced_mutation_count": 18,
                "native_attention_mutation_count": 18,
                "compiled_exact_39_event_control_executed": False,
                "model_forward_overlay_used_for_evaluation": True,
                "standalone_source_free_runtime_established": False,
                "deployed_candidate_artifact_materialized": False,
            },
            "explicit_distinction_from_weight_svd_ablation": {
                "previous_arm_id": "compressed_lexical_exact_body_ablation",
                "previous_arm_changed_only_tied_lexical_weights": True,
                "this_arm_keeps_lexical_boundary_exact": True,
                "this_arm_requires_prompt_nll_fisher": True,
                "this_arm_fits_activation_chart_routing": True,
                "this_arm_overlays_all_mlp_mutations_during_evaluation": True,
                "results_are_not_interchangeable": True,
            },
            "frozen_hyperparameters": {
                "chart_count": self.chart_count,
                "chart_rank": self.chart_rank,
                "generator_rank": self.generator_rank,
                "transport_rank": self.transport_rank,
                "maximum_incoming_charts_per_target": (
                    self.maximum_incoming_charts_per_target
                ),
                "batch_size": self.batch_size,
                "deterministic_seed": 1729,
                "affine_relative_ridge": self.affine_ridge,
                "local_chart_relative_ridge": self.chart_ridge,
                "transport_relative_ridge": self.transport_ridge,
                "chart_temperature": self.chart_temperature,
            },
            "delegated_runtime_binding": {
                "layer_count": 18,
                "hidden_width": 640,
                "mode_count": 2048,
                "chart_count": self.chart_count,
                "chart_coordinate_rank": self.chart_rank,
                "generator_rank": self.generator_rank,
                "square_transport_rank": self.transport_rank,
                "maximum_incoming_charts_per_target": (
                    self.maximum_incoming_charts_per_target
                ),
                "transport_scope": "adjacent_layer_same_token_only",
                "chart_and_rank_values_are_bound_by_transport_protocol": True,
            },
            "data_firewall": {
                "ordered_roles": ["A_fit", "A_eval"],
                "A_fit": (
                    "progressive Fisher capture, chart/generator fit, and "
                    "candidate-prefix transport fit"
                ),
                "A_eval": "prompt-disjoint development evaluation after fit",
                "prompt_and_example_id_disjoint_required": True,
                "development_examples_per_role": 5,
                "same_prompt_families_may_appear_in_both_roles": True,
                "family_disjoint": False,
                "heldout_confirmation_established": False,
                "A_eval_may_not_refit_any_tensor": True,
                "A_eval_rows_may_not_enter_capture_or_fit": True,
                "A_select_B_guard_C_assessment_used_by_default_worker": False,
            },
            "teacher_capture_contract": {
                "source_role": "A_fit",
                "objective": "summed_next_token_negative_log_likelihood",
                "fisher_estimator": (
                    "row_weight_from_squared_modal_state_times_"
                    "summed_NLL_score_gradient"
                ),
                "gradient_pooling": (
                    "all_attention_mask_rows_then_mean_over_modal_coordinates"
                ),
                "full_Fisher_coupling_matrix_computed": False,
                "outer_product_matrix_computed": False,
                "supervised_next_token_rows_only": False,
                "final_valid_token_zero_gradient_row_is_retained": True,
                "capture_fields": [
                    "normalized_feed_forward_inputs",
                    "native_down_projection_modal_states",
                    "score_gradients_wrt_modal_states",
                    "example_id_and_logical_position_row_keys",
                    "candidate_prefix_emitted_generator_latents",
                ],
                "row_fisher_normalization": "mean_one_clamped_0p1_to_10",
                "all_18_MLP_targets_captured_progressively": True,
                "raw_rows_remain_ephemeral": True,
                "row_identity_sha256_required": True,
            },
            "chart_and_generator_contract": {
                "atlas_input": (
                    "Fisher_weighted_normalized_feed_forward_input_rows"
                ),
                "chart_center_and_tangent_basis_required": True,
                "routing_input": "current_normalized_feed_forward_input",
                "generator_parameters_are_layer_specific": True,
                "cross_provenance_generator_families_discovered": False,
                "cross_provenance_parameter_sharing_enabled": False,
                "layer_ordinal_is_runtime_invocation_identity": True,
                "factorization": "per_chart_canonical_truncated_SVD",
            },
            "candidate_trajectory_transport_contract": {
                "progressive_prefix_frozen_before_next_layer_capture": True,
                "candidate_A_fit_prefix_trajectory_recaptured": True,
                "teacher_rows_may_not_substitute_for_candidate_source_states": True,
                "transport_target": (
                    "native_modal_state_minus_frozen_local_generator_modal_state"
                ),
                "fit_input": "candidate_prefix_emitted_generator_latents",
                "fit_output": "target_chart_modal_residual_correction",
                "causal_predecessor_only": True,
                "current_executor_edge_scope": "same_token_adjacent_layer_only",
                "transport_matrix_shape": "generator_rank_by_generator_rank",
                "maximum_incoming_charts_per_target": (
                    self.maximum_incoming_charts_per_target
                ),
            },
            "reference_transformer_schedule": {
                "events": events,
                "event_ids": [event["event_id"] for event in events],
                "event_count": len(events),
                "causal_order_is_strict": True,
                "compiled_as_independent_39_event_executor": False,
                "executed_through_native_model_forward_with_MLP_overlays": True,
                "native_tied_lexical_boundary_preserved": True,
                "tied_lexical_physical_storage_counted_once": True,
                "ingress_and_egress_lexical_weight_not_double_counted": True,
                "runtime_tied_storage_alias_reverified_by_worker": False,
                "native_attention_and_residual_path_preserved": True,
            },
            "default_worker_phase_contract": [
                {
                    "phase": "authenticate_setup_protocol_and_panel",
                    "consumes": ["setup", "transport_protocol", "panel"],
                },
                {
                    "phase": "load_local_model_and_tokenize_A_fit_A_eval",
                    "consumes": ["pinned_local_model", "A_fit", "A_eval"],
                },
                {
                    "phase": "fit_progressive_independent_chart_stack",
                    "consumes": ["A_fit", "native_MLP_targets"],
                },
                {
                    "phase": "fit_progressive_adjacent_transport_stack",
                    "consumes": ["A_fit", "candidate_prefix_trajectories"],
                },
                {
                    "phase": "fit_row_shuffled_transport_negative_control",
                    "consumes": ["A_fit", "shuffled_fit_only_source_packets"],
                },
                {
                    "phase": "evaluate_frozen_development_controls",
                    "consumes": ["A_eval", "three_frozen_stacks"],
                },
                {
                    "phase": "write_and_strictly_reload_wrapper_report",
                    "consumes": ["delegated_report", "setup_sha256"],
                },
            ],
            "actual_evaluation_conditions": [
                "native",
                "progressive_independent_factorized_chart",
                "progressive_factorized_chart_transport",
                "transport_zero_edge_control",
                "transport_row_shuffled_negative_control",
            ],
            "resource_and_claim_firewall": {
                "report_projected_unique_coefficient_scalars": True,
                "report_projected_float32_coefficient_bytes": True,
                "copied_exact_coefficients_must_be_counted": True,
                "tied_lexical_storage_must_be_counted_once": True,
                "overlay_runtime_is_not_source_free": True,
                "projected_accounting_is_not_resident_memory_measurement": True,
                "serialized_source_free_candidate_artifact_written": False,
                "setup_artifact_authorizes_no_success_claim": True,
                "bounded_mechanics_rung_compression_success_authorized": False,
                "maximum_zero_edge_projected_logical_parameter_reduction_fraction": (
                    0.07758274342008205
                ),
                "twenty_percent_whole_model_compression_gate_reachable": False,
                "latency_claim_without_measurement": False,
            },
            "deferred_from_bounded_arm": [
                "exact_39_event_control",
                "full_rank_chart_identity_control",
                "activation_only_no_Fisher_control",
                "matched_weight_SVD_body_control",
                "matched_deletion_control",
                "family_shuffled_control",
                "family_disjoint_guard_and_assessment",
                "cross_provenance_generator_family_discovery",
                "runtime cross-provenance generator-family parameter sharing",
                "compressed tied lexical boundary combined with chart body",
                "standalone source-free exact-attention executor",
                "all-attention chart replacement",
                "decode and KV-cache execution",
            ],
            "future_full_protocol_orchestration_seam": {
                "implemented_by_default_worker": False,
                "roles": ["A_fit", "A_select", "B_guard", "C_assessment"],
                "purpose": (
                    "injectable_test_seam_for_the_deferred_family_disjoint_"
                    "full_protocol_only"
                ),
            },
            "runnable_worker_entrypoint": (
                "fisher_graph.gemma3_fisher_activation_chart_whole_model_"
                "experiment:run_gemma3_fisher_activation_chart_whole_model_"
                "experiment"
            ),
        }
        state["setup_sha256"] = _sha256(state)
        return state


def build_fisher_activation_chart_whole_model_experiment_setup(
) -> dict[str, object]:
    return FisherActivationChartWholeModelExperimentSetup().state_dict()


def validate_fisher_activation_chart_whole_model_experiment_setup(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Fisher activation-chart setup must be a mapping")
    observed = json.loads(_canonical_json_bytes(dict(value)))
    expected = build_fisher_activation_chart_whole_model_experiment_setup()
    if observed != expected:
        raise ValueError("Fisher activation-chart setup does not match frozen v1")
    digest = observed.get("setup_sha256")
    without_digest = {
        key: item for key, item in observed.items() if key != "setup_sha256"
    }
    if digest != _sha256(without_digest):
        raise ValueError("Fisher activation-chart setup SHA-256 is invalid")
    return dict(observed)


def load_fisher_activation_chart_whole_model_experiment_setup(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Fisher activation-chart setup is invalid JSON") from error
    return validate_fisher_activation_chart_whole_model_experiment_setup(value)


def write_fisher_activation_chart_whole_model_experiment_setup(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(destination)
    payload = build_fisher_activation_chart_whole_model_experiment_setup()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write the setup-only contract for the runnable bounded Gemma "
            "Fisher/activation-chart whole-model worker."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate an existing setup artifact instead of writing it",
    )
    args = parser.parse_args(argv)
    if args.check:
        payload = load_fisher_activation_chart_whole_model_experiment_setup(
            args.output
        )
        action = "checked"
    else:
        payload = write_fisher_activation_chart_whole_model_experiment_setup(
            args.output
        )
        action = "written"
    print(
        json.dumps(
            {
                "action": action,
                "output": str(args.output),
                "setup_sha256": payload["setup_sha256"],
                "setup_only": True,
                "model_weights_loaded": False,
                "experiment_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID",
    "FISHER_ACTIVATION_CHART_SETUP_FORMAT_VERSION",
    "FISHER_ACTIVATION_CHART_SETUP_SCHEMA",
    "FisherActivationChartWholeModelExperimentSetup",
    "build_fisher_activation_chart_whole_model_experiment_setup",
    "load_fisher_activation_chart_whole_model_experiment_setup",
    "main",
    "validate_fisher_activation_chart_whole_model_experiment_setup",
    "write_fisher_activation_chart_whole_model_experiment_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
