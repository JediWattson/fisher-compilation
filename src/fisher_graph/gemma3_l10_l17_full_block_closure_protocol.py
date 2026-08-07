"""Frozen open-A protocol for Gemma Layer-17 full-block closure.

The preceding A3 trajectory experiment restored the native Layer-17 raw MLP
operator output on a Layer-10-compiled trajectory.  That target necessarily
leaves the post-attention residual-stream offset untouched.  Gemma 3 also
applies a post-feed-forward RMSNorm before the residual add, so a block-space
correction cannot be injected at A3's raw-MLP boundary.  This protocol changes
the target and explicitly relocates the generated contribution to the
post-feed-forward residual-delta boundary.  The four frozen decoder *tensors*
are reused there; their source bindings and artifact hashes are not silently
reinterpreted.

Everything else remains fixed: the authenticated A-fit corpus, eight outer
family folds, native virtual-gate Fisher weights, four affine decoder codecs,
rank-16 coordinate generators, ridge zero, the frozen Layer-10 graph, and the
stored-parameter/MAC envelope.  It authorizes adaptive A-fit development only;
selection, guard, Calibration-B, validation, test, and serving stay closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_l10_l17_trajectory_correction_protocol import (
    build_default_gemma3_l10_l17_trajectory_correction_protocol,
)


__all__ = [
    "FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_FORMAT_VERSION",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SCHEMA",
    "build_default_gemma3_l10_l17_full_block_closure_protocol",
    "validate_gemma3_l10_l17_full_block_closure_protocol",
]


GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_full_block_closure_protocol"
)
GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_FORMAT_VERSION = 1

_PROTOCOL_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-full-block-closure-protocol:v1\0"
)
_FOLD_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-full-block-closure-fold:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Filled only after the source-safe payload is reviewed.  ``build_default``
# refuses to operate if any field later drifts from this exact identity.
FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256 = (
    "9adefa7d75d11343d8ab103ac7c683aaea269f65b894b11039eaa508c08fa3dc"
)

_PRIOR_A3_EVIDENCE = {
    "report_file": (
        "layer10-layer17-a3-trajectory-correction-a-fit-lofo-v1.json"
    ),
    "report_file_sha256": (
        "916505564fbdbc25d4eed6c02ede508a19a776394dbe5e9d4539d634554310c9"
    ),
    "report_sha256": (
        "b90949076c560efa17181a5b47c6191d9358dd5b3d6b5c4e2ba57d6937bdd9ca"
    ),
    "protocol_sha256": (
        "ab3794c3cf6660738db6b24c66db02383a72d932e0b540462cec8fa41aff55e3"
    ),
    "all_required_gates_pass": False,
    "next_action": "stop_keep_other_roles_closed_and_revise_a_fit_recipe",
    "eligible_condition": "trajectory_corrected_composition",
    "family_macro_delta_nll_per_token": 0.0973176092618302,
    "family_macro_native_to_candidate_kl_per_token": 0.1183239227142372,
    "family_macro_top1_agreement_to_native": 0.8148135953989659,
}

_EDGELESS_DIAGNOSTIC_RESOURCES = {
    "replaced_layer_count": 2,
    "graph_node_count": 8,
    "interaction_count": 0,
    "native_removed_parameters": 1_082_880,
    "graph_parameters": 290_710,
    "net_parameter_savings": 792_170,
    "candidate_whole_model_learned_parameters": 267_306_006,
    "dense_graph_macs_per_token": 285_280,
    "executed_graph_macs_per_token": 285_280,
    "net_executed_macs_saved_per_token": 797_600,
}


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("canonical JSON does not permit non-finite values")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_canonical_json_value(item) for item in value]
    raise TypeError("protocols must contain only strict JSON values")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _folds(source: Mapping[str, object]) -> list[dict[str, object]]:
    raw_folds = source.get("folds")
    if isinstance(raw_folds, (str, bytes)) or not isinstance(
        raw_folds,
        Sequence,
    ):
        raise RuntimeError("source A3 folds are unavailable")
    result: list[dict[str, object]] = []
    for raw in raw_folds:
        if not isinstance(raw, Mapping):
            raise RuntimeError("source A3 fold is invalid")
        payload = {
            key: raw[key]
            for key in (
                "fold_index",
                "fold_id",
                "held_family_alias",
                "training_family_aliases",
                "held_example_count",
                "training_example_count",
                "held_membership_sha256",
                "training_membership_sha256",
                "fit_role_manifest_sha256",
            )
        }
        payload.update(
            {
                "fit_policy": (
                    "fit_a4_full_block_closure_on_training_complement_only"
                ),
                "score_policy": (
                    "score_frozen_a3_a4_and_interaction_factorial_conditions_"
                    "once_on_held_family"
                ),
            }
        )
        result.append(
            {
                **payload,
                "artifact_sha256": _domain_sha256(_FOLD_DOMAIN, payload),
            }
        )
    if len(result) != 8:
        raise RuntimeError("full-block closure requires exactly eight folds")
    return result


def _protocol_payload() -> dict[str, object]:
    source = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    source_authority = dict(source["source_authority"])
    source_authority["prior_a3_evidence"] = dict(_PRIOR_A3_EVIDENCE)

    projection = dict(source["projection_contract"])
    projection.update(
        {
            "projection_id": (
                "a4-full-block-concatenated-frozen-affine-decoder-sum-v1"
            ),
            "projection_definition": (
                "minimum_norm_coordinates_for_full_block_closure_target_in_"
                "affine_sum_of_frozen_mode_codecs"
            ),
            "projection_formula": (
                "c_star=(g_block-mu_cat)*D_cat_T*pinv(D_cat*D_cat_T)"
            ),
            "reconstruction_formula": "g_projected=mu_cat+c_star*D_cat",
            "source_decoder_output_boundary": (
                "layer.17.mlp.operator_output"
            ),
            "runtime_decoder_output_boundary": "layer.17.mlp.delta",
            "decoder_tensor_values_preserved_exactly": True,
            "source_basis_bindings_preserved": False,
            "relocated_basis_binding_required": True,
            "relocated_basis_artifact_sha256_recorded_per_fold": True,
        }
    )

    candidate = dict(source["candidate_contract"])
    candidate.update(
        {
            "candidate_id": "l10-frozen-l17-a4-full-block-c48-r16-v1",
            "learned_capacity_unchanged_from_a3": True,
            "runtime_application_boundary_changed_from_a3": True,
            "runtime_application_boundary": "layer.17.mlp.delta",
            "compact_path": (
                "compact_raw_mlp_then_live_post_feedforward_rmsnorm"
            ),
            "generated_path": (
                "add_graph_closure_after_live_post_feedforward_rmsnorm"
            ),
            "source_decoder_tensor_values_preserved_exactly": True,
            "fold_executable_bundle_required": True,
            "fold_executable_bundle_contains_rows_or_prompts": False,
        }
    )

    gates = dict(source["gates"])
    gates.update(
        {
            "require_strict_family_macro_kl_improvement_vs_prior_a3": True,
            "minimum_held_family_kl_improvement_count_vs_prior_a3": 6,
            "prior_a3_report_sha256": _PRIOR_A3_EVIDENCE["report_sha256"],
            "require_post_feedforward_application_boundary": True,
            "require_full_block_capture_audits": True,
            "require_fold_executable_bundle": True,
        }
    )

    return {
        "schema": GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_FORMAT_VERSION
        ),
        "source_authority": source_authority,
        "target_contract": {
            "target_variant": "A4_full_block_closure",
            "layer_ordinal": 17,
            "pairing_key": ["example_id", "logical_position"],
            "pairing_requires_identical_token_stream": True,
            "generator_input": (
                "layer10_compiled_layer17_mlp_normalized_input"
            ),
            "native_pass": "fully_native_model",
            "native_post_attention_capture": "layer.17.post_attention",
            "native_mlp_capture": "layer.17.mlp.operator_output",
            "native_post_feedforward_delta_capture": "layer.17.mlp.delta",
            "native_block_output_capture": "layer.17.output",
            "compiled_pass": "frozen_layer10_generated_graph_overlay",
            "compiled_post_attention_capture": "layer.17.post_attention",
            "compiled_mlp_capture": "layer.17.mlp.operator_output",
            "compiled_post_feedforward_delta_capture": "layer.17.mlp.delta",
            "compiled_block_output_capture": "layer.17.output",
            "compiled_keep_pass": (
                "authenticated_layer17_compact_raw_mlp_replay_on_captured_"
                "layer10_compiled_normalized_input_then_live_"
                "post_feedforward_rmsnorm"
            ),
            "compiled_keep_capture": (
                "exact_compact_retained_layer17_post_feedforward_delta"
            ),
            "compiled_keep_replay_uses_source_model_dtype_and_device": True,
            "compiled_keep_replay_matches_runtime_deletion_operator": True,
            "raw_target_symbol": "g_block",
            "raw_target_formula": (
                "native_layer17_block_output-"
                "compiled_layer17_post_attention_residual-"
                "compiled_keep_layer17_post_feedforward_delta"
            ),
            "equivalent_formula": (
                "native_post_feedforward_delta-"
                "compiled_keep_post_feedforward_delta+"
                "native_post_attention_residual-"
                "compiled_post_attention_residual"
            ),
            "target_width": 640,
            "includes_layer17_residual_stream_offset": True,
            "generator_application_boundary": "layer.17.mlp.delta",
            "pre_norm_target_inversion_forbidden": True,
            "reason_pre_norm_target_inversion_forbidden": (
                "rmsnorm_is_nonlinear_and_not_generally_invertible"
            ),
            "native_block_decomposition_audit_required": True,
            "compiled_block_decomposition_audit_required": True,
            "a4_minus_native_delta_closure_offset_identity_audit_required": (
                True
            ),
            "decomposition_audit_numeric_model": (
                "float32_residual_addition_roundoff_with_relative_guard"
            ),
            "maximum_decomposition_max_abs_difference": 0.01,
            "maximum_decomposition_rmse": 0.0002,
            "maximum_decomposition_normalized_rmse": 0.000001,
            "algebraic_full_minus_selected_used_as_target": False,
            "algebraic_full_minus_selected_equivalence_audit_required": True,
            "compact_equivalence_audit_numeric_model": (
                "post_rmsnorm_amplification_with_relative_guard"
            ),
            "maximum_compact_equivalence_max_abs_difference": 0.05,
            "maximum_compact_equivalence_rmse": 0.002,
            "maximum_compact_equivalence_normalized_rmse": 0.00001,
            "fisher_weight_source": source["target_contract"][
                "fisher_weight_source"
            ],
            "fisher_weight_formula": source["target_contract"][
                "fisher_weight_formula"
            ],
            "fisher_activation_site": source["target_contract"][
                "fisher_activation_site"
            ],
            "fisher_family_normalization": source["target_contract"][
                "fisher_family_normalization"
            ],
            "activation_rows_are_ephemeral": True,
        },
        "projection_contract": projection,
        "candidate_contract": candidate,
        "folds": _folds(source),
        "gates": gates,
        "evaluation_contract": {
            "outer_split_unit": (
                "authenticated_calibration_a_fit_family_alias"
            ),
            "outer_fold_count": 8,
            "training_family_count_per_fold": 7,
            "held_family_count_per_fold": 1,
            "fold_local_projection_and_refit_required": True,
            "held_family_used_for_projection_fit_or_early_stopping": False,
            "arm_or_target_selection_inside_lofo": False,
            "aggregation": "unweighted_macro_mean_over_held_families",
            "conditions": [
                "native",
                "layer10_only",
                "source_layer17_only",
                "a4_full_block_layer17_only",
                "frozen_uncorrected_composition",
                "a4_full_block_corrected_composition",
                "l10_edgeless_frozen_l17_composition",
                "l10_edgeless_a4_composition",
                "matched_double_deletion",
            ],
            "eligible_condition": "a4_full_block_corrected_composition",
            "comparison_condition": "frozen_uncorrected_composition",
            "prior_a3_comparison_condition": (
                "trajectory_corrected_composition"
            ),
            "family_kl_improvement_definition": (
                "a4_full_block_corrected_composition_kl<"
                "frozen_uncorrected_composition_kl"
            ),
            "interaction_factorial": {
                "frozen_edges_off": (
                    "l10_edgeless_frozen_l17_composition"
                ),
                "frozen_edges_on": "frozen_uncorrected_composition",
                "a4_edges_off": "l10_edgeless_a4_composition",
                "a4_edges_on": "a4_full_block_corrected_composition",
                "difference_in_differences": (
                    "(a4_edges_on-a4_edges_off)-"
                    "(frozen_edges_on-frozen_edges_off)"
                ),
                "diagnostic_only": True,
                "may_select_or_mutate_interactions": False,
            },
            "deletion_nll_recovery": {
                "candidate_condition": (
                    "a4_full_block_corrected_composition"
                ),
                "control_condition": "matched_double_deletion",
                "formula": (
                    "(matched_double_deletion_delta_nll-"
                    "a4_full_block_corrected_composition_delta_nll)/"
                    "matched_double_deletion_delta_nll"
                ),
                "denominator_requirement": "strictly_greater_than_zero",
                "invalid_denominator_policy": "fail_closed",
                "macro_aggregation": "equal_family_arithmetic_mean",
                "worst_aggregation": "minimum_over_held_families",
            },
            "primary_resources": candidate["exact_resources"],
            "edgeless_diagnostic_resources": dict(
                _EDGELESS_DIAGNOSTIC_RESOURCES
            ),
            "resource_accounting_replayed_from_exact_graph": True,
            "projection_receipt_required_per_fold": True,
            "randomness_policy": "seed_and_recipe_committed_before_execution",
        },
        "claim_boundary": {
            "scientific_role": (
                "calibration_a_fit_adaptive_full_block_closure_development"
            ),
            "supports_family_blocked_internal_estimate": True,
            "supports_heldout_confirmation": False,
            "supports_serving_authorization": False,
            "supports_whole_model_compiled_claim": False,
            "full_model_logits_may_be_scored": True,
            "compiled_layer_count": 2,
            "supports_lossless_claim": False,
            "supports_latency_or_kernel_speed_claim": False,
            "resource_claim_limited_to_exact_static_accounting": True,
            "selection_guard_b_validation_and_test_remain_untouched": True,
        },
        "safety": dict(source["safety"]),
    }


def build_default_gemma3_l10_l17_full_block_closure_protocol(
) -> dict[str, object]:
    """Return the exact source-safe A4 fit-only protocol."""

    payload = _protocol_payload()
    artifact_sha256 = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if (
        artifact_sha256
        != FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256
    ):
        raise RuntimeError("frozen full-block closure protocol drifted")
    return json.loads(
        _canonical_json_bytes(
            {**payload, "artifact_sha256": artifact_sha256}
        ).decode("utf-8")
    )


def validate_gemma3_l10_l17_full_block_closure_protocol(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed unless *raw* is the exact frozen A4 protocol."""

    if not isinstance(raw, Mapping):
        raise TypeError("full-block closure protocol must be a mapping")
    expected_fields = {*_protocol_payload(), "artifact_sha256"}
    if set(raw) != expected_fields:
        raise ValueError("full-block closure protocol fields are invalid")
    supplied = raw.get("artifact_sha256")
    if not isinstance(supplied, str) or _SHA256.fullmatch(supplied) is None:
        raise ValueError("full-block closure protocol hash is invalid")
    payload = {key: raw[key] for key in raw if key != "artifact_sha256"}
    if _canonical_json_value(payload) != _canonical_json_value(
        _protocol_payload()
    ):
        raise ValueError("full-block closure protocol differs from frozen plan")
    recomputed = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if supplied != recomputed:
        raise ValueError("full-block closure protocol hash mismatch")
    if supplied != FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256:
        raise ValueError("full-block closure protocol identity is not frozen")
    return json.loads(_canonical_json_bytes(raw).decode("utf-8"))
