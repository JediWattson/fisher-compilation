"""Frozen fit-only protocol for Gemma layer-10 -> layer-17 correction.

This module is deliberately a pure, source-safe planner.  It opens no file,
loads no model, imports no tensor runtime, and reads no metric.  The artifact
freezes the first composition-correction arm before any new activation row is
collected:

* the exact qualified layer-10 graph from the current adaptive composition;
* eight outer LOFO folds from v8 ``calibration_a_fit`` only;
* the A3 raw Layer-17 MLP target on the layer-10-compiled trajectory;
* one joint projection into the affine sum of the frozen decoder codecs; and
* four independent cap-48/rank-16 generator fits after a fixed coordinate
  split, with exactly the current two-layer resource envelope.

The resulting protocol can authorize an A-fit adaptive-development diagnostic
only.  Calibration-A selection, guard, Calibration-B, validation, and test are
outside this artifact's authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    build_default_v8_layer17_family_lofo_protocol,
)


__all__ = [
    "FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256",
    "GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_FORMAT_VERSION",
    "GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SCHEMA",
    "build_default_gemma3_l10_l17_trajectory_correction_protocol",
    "validate_gemma3_l10_l17_trajectory_correction_protocol",
]


GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_trajectory_correction_protocol"
)
GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_FORMAT_VERSION = 2

_PROTOCOL_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-trajectory-correction-protocol:v2\0"
)
_FOLD_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-trajectory-correction-fold:v1\0"
)
_DECODER_SPAN_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-trajectory-affine-decoder-sum:v2\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Validation requires this exact identity.  Any target, gate, resource, basis,
# or authority change therefore requires a new protocol version.
FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256 = (
    "ab3794c3cf6660738db6b24c66db02383a72d932e0b540462cec8fa41aff55e3"
)

_SUMMED_MEAN_SHA256 = (
    "f906a94712c4026c03a4eb5aeb62ea188d5987ccdaeba9e5fb6ab4492df9dc1e"
)

_MODEL_AUTHORITY = {
    "model_id": "google/gemma-3-270m",
    "requested_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "adapter_model_fingerprint": (
        "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
    ),
}

_COMPOSITION_AUTHORITY = {
    "tensor_file": "layer10-layer17-adaptive-composition-open-a-v2.pt",
    "tensor_file_sha256": (
        "394906f8e84a50e18922de0dc8c114be1ea9889f0995ccca180b9f6a8d303d8d"
    ),
    "composition_payload_sha256": (
        "2f7c2179656fc16c614cd84b7a0b29d3250443a5d8c80db221b220e3d3f082bf"
    ),
    "combined_edgeless_graph_sha256": (
        "76e6ca06124a542e0f1ce4b26315f5892abdb1057d07d097a849dca312dc3f6c"
    ),
    "combined_primary_graph_sha256": (
        "35d35f2318e0728bb649f2825601d2edbe06e13307a412c4d81c66e8e387c4ca"
    ),
}

_LAYER10_AUTHORITY = {
    "candidate_tensor_file": "layer10-shape-flow-gain-dev-v2.pt",
    "candidate_tensor_file_sha256": (
        "feffc023ba37aee10591cc4313238dd6936181a5e77c5a61d12cfe6be04b8a1b"
    ),
    "candidate_scientific_payload_sha256": (
        "eae90f334b34dc76d7ef38585e394f77825e1514189cecc3e38e36ec3842fcbb"
    ),
    "primary_graph_sha256": (
        "67327f1ba3cff3bd9a49897245d0301d109ac1564eff2c4f70409d29a28a8b94"
    ),
    "node_count": 4,
    "interaction_count": 3,
    "graph_parameter_count": 132_035,
    "dense_graph_macs_per_token": 129_248,
    "executed_graph_macs_per_token": 126_432,
    "must_remain_bitwise_unchanged": True,
}

_LAYER17_SOURCE_AUTHORITY = {
    "candidate_tensor_file": (
        "layer17-capped-node-c48-r16-edgeless-a-fit-v8-full-refit-dev-v1.pt"
    ),
    "candidate_tensor_file_sha256": (
        "fc989138da2c190c848fe64460752711b19144b68b20fedb047f6352e9aeea17"
    ),
    "candidate_scientific_payload_sha256": (
        "e0969e90e78c714dc27bc1ee80d925e4dddc02a6e3fff2ea610bd46815c7231e"
    ),
    "source_edgeless_graph_sha256": (
        "4b81283db0df73b3be06d67ed61be4733190824687b0d34a5d9b3662a26d1607"
    ),
    "fit_receipt_sha256": (
        "d20fde2c3263c8a5607da29e541d523f353de0683c7c85b138f6fcc1cc732756"
    ),
}

_DECODER_RECORDS = (
    {
        "node_name": (
            "gemma3.layer-17.cluster-0.modal-generator.same-layer-0.graph-node"
        ),
        "fragment_id": "cluster.0/layer.17",
        "mode_set_id": "cluster.0/layer.17",
        "node_rank": 48,
        "coordinate_start": 0,
        "coordinate_stop": 48,
        "computational_mode_basis_sha256": (
            "6ec0b3a6f62b21b983171970bee9f82ae5759884b4f534845a471ab884ccc1b0"
        ),
        "mean_bias_sha256": (
            "4e6e6690b5a82c183b483c676f47bd5ce1e819e0eef007265a5ca81f50814015"
        ),
        "decoder_basis_sha256": (
            "e73b3f8937b0193b55e310bfad3af072a7d00d39e41b9eaf90ea142d816573c1"
        ),
    },
    {
        "node_name": (
            "gemma3.layer-17.cluster-28.modal-generator.same-layer-1.graph-node"
        ),
        "fragment_id": "cluster.28/layer.17",
        "mode_set_id": "cluster.28/layer.17",
        "node_rank": 38,
        "coordinate_start": 48,
        "coordinate_stop": 86,
        "computational_mode_basis_sha256": (
            "f81f6943d66197825af04c856895b1be3af9c147191f535ffd82b393bde38ae1"
        ),
        "mean_bias_sha256": (
            "31ab398d5a7219575d2f554ef87b34dbf59d8705f55c33384673c725e42edb67"
        ),
        "decoder_basis_sha256": (
            "45564552ea5728d159dc887ff501dda20e7609ff9c469fdaf0994790ab208bda"
        ),
    },
    {
        "node_name": (
            "gemma3.layer-17.cluster-34.modal-generator.same-layer-2.graph-node"
        ),
        "fragment_id": "cluster.34/layer.17",
        "mode_set_id": "cluster.34/layer.17",
        "node_rank": 48,
        "coordinate_start": 86,
        "coordinate_stop": 134,
        "computational_mode_basis_sha256": (
            "21ef92ceeb63bd56692894dd6a82d852684c6c34798ada167a73b022e057bb11"
        ),
        "mean_bias_sha256": (
            "264330b03a9d11dca8ecf2ec89a4dcd8bb55ee214bc4a01cda85a4fd3a7b2086"
        ),
        "decoder_basis_sha256": (
            "47fcb24f6f73e9ec254248cd3795361106ad23680d0a94a603f6433032c3ade1"
        ),
    },
    {
        "node_name": (
            "gemma3.layer-17.cluster-54.modal-generator.same-layer-3.graph-node"
        ),
        "fragment_id": "cluster.54/layer.17",
        "mode_set_id": "cluster.54/layer.17",
        "node_rank": 48,
        "coordinate_start": 134,
        "coordinate_stop": 182,
        "computational_mode_basis_sha256": (
            "c77e9924c4b9600da7301fc32d684490fbfc7c21e0d7ec65559bbbb27b639c36"
        ),
        "mean_bias_sha256": (
            "2506ce745cfb5121ec8d43d830f368ad4dbf74da41591cf54f9a7b9f8cb99413"
        ),
        "decoder_basis_sha256": (
            "1fd48525f312702c3660c8566e1358c1634b8fff7117b3715d8f035f38cfcf70"
        ),
    },
)

_EXACT_RESOURCES = {
    "source_whole_model_learned_parameters": 268_098_176,
    "replaced_layer_count": 2,
    "graph_node_count": 8,
    "dynamic_interaction_count": 3,
    "native_removed_parameters": 1_082_880,
    "primary_graph_parameters": 295_129,
    "net_stored_parameter_savings": 787_751,
    "candidate_whole_model_learned_parameters": 267_310_425,
    "dense_graph_macs_per_token": 289_600,
    "executed_graph_macs_per_token": 286_784,
    "native_removed_macs_per_token": 1_082_880,
    "net_executed_macs_saved_per_token": 796_096,
}

_PROTOCOL_FIELDS = {
    "schema",
    "format_version",
    "source_authority",
    "target_contract",
    "projection_contract",
    "candidate_contract",
    "folds",
    "gates",
    "evaluation_contract",
    "claim_boundary",
    "safety",
    "artifact_sha256",
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
        value, (str, bytes, bytearray)
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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _decoder_span_sha256() -> str:
    return _domain_sha256(
        _DECODER_SPAN_DOMAIN,
        {
            "geometry": "affine_sum_of_frozen_mode_codecs",
            "concatenation_axis": "coordinate_rows",
            "ordered_decoders": list(_DECODER_RECORDS),
            "concatenated_decoder_shape": [182, 640],
            "summed_mean_shape": [640],
            "summed_mean_sha256": _SUMMED_MEAN_SHA256,
        },
    )


def _folds() -> list[dict[str, object]]:
    source = build_default_v8_layer17_family_lofo_protocol()
    source_folds = source["folds"]
    if not isinstance(source_folds, list) or len(source_folds) != 8:
        raise RuntimeError("frozen source LOFO folds drifted")
    folds: list[dict[str, object]] = []
    for raw in source_folds:
        if not isinstance(raw, Mapping):
            raise RuntimeError("frozen source LOFO fold is invalid")
        payload = {
            "fold_index": raw["fold_index"],
            "fold_id": raw["fold_id"],
            "held_family_alias": raw["held_family_alias"],
            "training_family_aliases": raw["training_family_aliases"],
            "held_example_count": raw["held_example_count"],
            "training_example_count": raw["training_example_count"],
            "held_membership_sha256": raw["held_membership_sha256"],
            "training_membership_sha256": raw["training_membership_sha256"],
            "fit_role_manifest_sha256": raw["fit_role_manifest_sha256"],
            "fit_policy": (
                "fit_a3_trajectory_correction_on_training_complement_only"
            ),
            "score_policy": (
                "score_frozen_uncorrected_and_corrected_compositions_once_"
                "on_held_family"
            ),
        }
        folds.append(
            {
                **payload,
                "artifact_sha256": _domain_sha256(_FOLD_DOMAIN, payload),
            }
        )
    return folds


def _source_fit_authority() -> dict[str, object]:
    source = build_default_v8_layer17_family_lofo_protocol()
    corpus = source["corpus_authority"]
    roles = source["role_bindings"]
    if not isinstance(corpus, Mapping) or not isinstance(roles, Mapping):
        raise RuntimeError("frozen source LOFO authority drifted")
    fit = roles.get("fit")
    if not isinstance(fit, Mapping):
        raise RuntimeError("frozen source A-fit authority drifted")
    return {
        "source_lofo_protocol_sha256": (
            FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        ),
        "corpus_schema": corpus["schema"],
        "corpus_id": corpus["corpus_id"],
        "corpus_artifact_sha256": corpus["artifact_sha256"],
        "tokenizer_contract_sha256": corpus["tokenizer_contract_sha256"],
        "fit_role": fit["role"],
        "fit_manifest_sha256": fit["manifest_sha256"],
        "fit_source_file_sha256": fit["source_file_sha256"],
        "fit_membership_sha256": corpus["fit_membership_sha256"],
        "family_alias_mapping_sha256": corpus[
            "family_alias_mapping_sha256"
        ],
        "family_aliases": list(V8_FAMILY_LOFO_FAMILY_ALIASES),
        "example_count": fit["example_count"],
        "family_count": fit["family_count"],
        "exclusive_role_use": "calibration_a_fit",
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_FORMAT_VERSION
        ),
        "source_authority": {
            "model": dict(_MODEL_AUTHORITY),
            "composition_bundle": dict(_COMPOSITION_AUTHORITY),
            "layer10": dict(_LAYER10_AUTHORITY),
            "layer17_decoder_source": dict(_LAYER17_SOURCE_AUTHORITY),
            "calibration_a_fit": _source_fit_authority(),
            "forbidden_roles": [
                "calibration_a_selection",
                "calibration_a_guard",
                "calibration_b",
                "validation",
                "test",
            ],
        },
        "target_contract": {
            "target_variant": "A3",
            "layer_ordinal": 17,
            "pairing_key": ["example_id", "logical_position"],
            "pairing_requires_identical_token_stream": True,
            "input": "layer10_compiled_layer17_mlp_normalized_input",
            "native_pass": "fully_native_model",
            "native_capture": "layer17_mlp_operator_output",
            "compiled_keep_pass": (
                "authenticated_layer17_compact_mlp_replay_on_captured_"
                "layer10_compiled_normalized_input"
            ),
            "compiled_keep_capture": (
                "exact_compact_retained_layer17_mlp_operator_output"
            ),
            "compiled_keep_replay_uses_source_model_dtype_and_device": True,
            "compiled_keep_replay_matches_runtime_deletion_operator": True,
            "algebraic_full_minus_selected_used_as_target": False,
            "algebraic_full_minus_selected_equivalence_audit_required": True,
            "maximum_algebraic_equivalence_max_abs_difference": 0.0001,
            "maximum_algebraic_equivalence_rmse": 0.00001,
            "raw_target_symbol": "r_star",
            "raw_target_formula": (
                "native_full_layer17_mlp_operator_output-"
                "compiled_keep_layer17_mlp_operator_output"
            ),
            "target_width": 640,
            "includes_layer17_residual_stream_offset": False,
            "a3_plus_is_forbidden": True,
            "fisher_weight_source": (
                "fully_native_selected_mode_virtual_gate_empirical_fisher_"
                "at_layer17_mlp_down_input"
            ),
            "fisher_weight_formula": (
                "sum_over_selected_modes((z*dNLL_dz)^2)"
            ),
            "fisher_activation_site": "layer.17.mlp.down_input",
            "fisher_family_normalization": (
                "equal_total_mass_per_training_family"
            ),
            "activation_rows_are_ephemeral": True,
        },
        "projection_contract": {
            "projection_id": "a3-concatenated-frozen-affine-decoder-sum-v2",
            "geometry": "affine_sum_of_frozen_mode_codecs",
            "decoder_concatenation_axis": "coordinate_rows",
            "ordered_decoders": [dict(row) for row in _DECODER_RECORDS],
            "concatenated_decoder_shape": [182, 640],
            "concatenated_coordinate_width": 182,
            "decoder_span_sha256": _decoder_span_sha256(),
            "affine_offset_definition": "mu_cat=sum_i(frozen_mean_bias_i)",
            "summed_mean_shape": [640],
            "summed_mean_sha256": _SUMMED_MEAN_SHA256,
            "summed_mean_tensor_hash_domain": (
                "fisher_graph.computational_modes.tensor.v1"
            ),
            "projection_definition": (
                "minimum_norm_coordinates_for_raw_target_in_affine_sum_of_"
                "frozen_mode_codecs"
            ),
            "projection_formula": (
                "c_star=(r_star-mu_cat)*D_cat_T*pinv(D_cat*D_cat_T)"
            ),
            "reconstruction_formula": (
                "r_projected=mu_cat+c_star*D_cat"
            ),
            "accumulation_dtype": "float64",
            "pseudoinverse_solver": "svd_moore_penrose",
            "singular_value_keep_rule": (
                "sigma>sigma_max*max(182,640)*float64_machine_epsilon"
            ),
            "coordinate_split_policy": (
                "contiguous_slices_in_frozen_decoder_order"
            ),
            "coordinate_slices": [
                {
                    "node_name": row["node_name"],
                    "start": row["coordinate_start"],
                    "stop": row["coordinate_stop"],
                }
                for row in _DECODER_RECORDS
            ],
            "joint_projection_required": True,
            "independent_generator_fits_after_split": True,
            "direct_independent_decoder_projection_forbidden": True,
            "projection_metadata_must_match_exactly": True,
        },
        "candidate_contract": {
            "candidate_id": "l10-frozen-l17-a3-c48-r16-edgeless-v1",
            "layer10_policy": "reuse_exact_frozen_primary_graph",
            "layer17_fragment_ids_in_execution_order": [
                row["fragment_id"] for row in _DECODER_RECORDS
            ],
            "layer17_node_ranks_in_execution_order": [48, 38, 48, 48],
            "layer17_mode_rank_cap": 48,
            "layer17_generator_rank": 16,
            "layer17_ridge": 0.0,
            "layer17_edge_policy": "edgeless",
            "layer17_interaction_count": 0,
            "frozen_decoder_bases_required": True,
            "frozen_decoder_means_required": True,
            "decoder_basis_refit_forbidden": True,
            "joint_coordinate_target_then_independent_generator_fit": True,
            "fold_local_coefficient_refit_required": True,
            "reuse_fitted_coefficients_across_folds": False,
            "source_model_mutation_forbidden": True,
            "layer10_mutation_forbidden": True,
            "exact_resources": dict(_EXACT_RESOURCES),
        },
        "folds": _folds(),
        "gates": {
            "decision_policy": "all_required_gates_must_pass",
            "required_completed_fold_count": 8,
            "maximum_failed_fold_count": 0,
            "maximum_family_macro_delta_nll_per_token": 0.08,
            "maximum_worst_family_delta_nll_per_token": 0.10,
            "maximum_family_macro_native_to_candidate_kl_per_token": 0.09,
            "minimum_family_macro_top1_agreement_to_native": 0.84,
            "maximum_family_macro_interaction_excess_nll": 0.01,
            "require_strict_family_macro_kl_improvement_vs_frozen": True,
            "minimum_held_family_kl_improvement_count": 6,
            "required_held_family_count": 8,
            "maximum_family_macro_nll_regression_vs_frozen": 0.001,
            "maximum_family_macro_top1_regression_vs_frozen": 0.005,
            "minimum_family_macro_deletion_nll_recovery_fraction": 0.60,
            "minimum_worst_family_deletion_nll_recovery_fraction": 0.40,
            "require_exact_resources": True,
            "require_exact_projection_metadata": True,
            "require_compact_replay_algebraic_equivalence_audit": True,
            "require_source_model_unchanged": True,
            "require_layer10_unchanged": True,
            "permit_latency_or_kernel_speed_claim": False,
        },
        "evaluation_contract": {
            "outer_split_unit": "authenticated_calibration_a_fit_family_alias",
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
                "trajectory_corrected_layer17_only",
                "frozen_uncorrected_composition",
                "trajectory_corrected_composition",
                "matched_double_deletion",
            ],
            "eligible_condition": "trajectory_corrected_composition",
            "comparison_condition": "frozen_uncorrected_composition",
            "family_kl_improvement_definition": (
                "trajectory_corrected_composition_kl<"
                "frozen_uncorrected_composition_kl"
            ),
            "interaction_excess_nll_formula": (
                "corrected_composition_delta_nll-layer10_only_delta_nll-"
                "corrected_layer17_only_delta_nll"
            ),
            "deletion_nll_recovery": {
                "candidate_condition": "trajectory_corrected_composition",
                "control_condition": "matched_double_deletion",
                "formula": (
                    "(matched_double_deletion_delta_nll-"
                    "trajectory_corrected_composition_delta_nll)/"
                    "matched_double_deletion_delta_nll"
                ),
                "denominator_requirement": "strictly_greater_than_zero",
                "invalid_denominator_policy": "fail_closed",
                "macro_aggregation": "equal_family_arithmetic_mean",
                "worst_aggregation": "minimum_over_held_families",
            },
            "resource_accounting_replayed_from_exact_graph": True,
            "projection_receipt_required_per_fold": True,
            "randomness_policy": "seed_and_recipe_committed_before_execution",
        },
        "claim_boundary": {
            "scientific_role": (
                "calibration_a_fit_adaptive_trajectory_correction_development"
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
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_per_example_identity": False,
            "contains_activation_gradient_or_projection_tensors": False,
            "contains_model_or_candidate_weights": False,
            "filesystem_accessed": False,
            "role_input_file_opened": False,
            "model_or_tokenizer_accessed": False,
            "model_executed": False,
            "metrics_read": False,
            "calibration_a_fit_opened_by_planner": False,
            "calibration_a_selection_opened": False,
            "calibration_a_guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
    }


def build_default_gemma3_l10_l17_trajectory_correction_protocol(
) -> dict[str, object]:
    """Return the exact source-safe A3 fit-only protocol."""

    payload = _protocol_payload()
    artifact_sha256 = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if (
        artifact_sha256
        != FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256
    ):
        raise RuntimeError("frozen trajectory-correction protocol drifted")
    return json.loads(
        _canonical_json_bytes(
            {**payload, "artifact_sha256": artifact_sha256}
        ).decode("utf-8")
    )


def validate_gemma3_l10_l17_trajectory_correction_protocol(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed unless *raw* is the exact frozen A3 protocol."""

    protocol = _strict_mapping(
        raw,
        fields=_PROTOCOL_FIELDS,
        label="trajectory-correction protocol",
    )
    supplied_sha256 = _require_sha256(
        protocol["artifact_sha256"],
        label="trajectory-correction protocol",
    )
    payload = {
        key: protocol[key]
        for key in _PROTOCOL_FIELDS
        if key != "artifact_sha256"
    }
    if _canonical_json_value(payload) != _canonical_json_value(
        _protocol_payload()
    ):
        raise ValueError(
            "trajectory-correction protocol differs from frozen plan"
        )
    recomputed_sha256 = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if supplied_sha256 != recomputed_sha256:
        raise ValueError("trajectory-correction protocol hash mismatch")
    if (
        supplied_sha256
        != FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256
    ):
        raise ValueError("trajectory-correction protocol identity is not frozen")
    return json.loads(_canonical_json_bytes(protocol).decode("utf-8"))
