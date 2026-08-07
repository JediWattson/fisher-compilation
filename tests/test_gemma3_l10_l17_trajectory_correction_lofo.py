from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l10_l17_trajectory_correction_lofo as lofo
import fisher_graph.gemma3_layer17_family_lofo_authority as authority_module
from fisher_graph.gemma3_l10_l17_trajectory_correction_protocol import (
    build_default_gemma3_l10_l17_trajectory_correction_protocol,
)
from fisher_graph.gemma3_layer17_family_lofo_authority import (
    GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
    GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA,
)
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    build_default_v8_layer17_family_lofo_protocol,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import LayerFragmentRows
from fisher_graph.gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


def _metric(native: float, delta: float, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _evaluation(index: int = 0) -> dict[str, object]:
    native = 2.0 + index * 0.001
    resources = {
        condition: {
            **values,
            "executed_peak_live_modal_width": (
                0 if condition == "matched_double_deletion" else 48
            ),
        }
        for condition, values in lofo._EXPECTED_CONDITION_RESOURCES.items()
    }
    return {
        "execution_path": "full_model_logits_fixed_capacity_a3_trajectory_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": 100,
        "logical_valid_tokens": 128,
        "native": {"nll_per_token": native},
        "conditions": {
            "layer10_only": _metric(native, 0.010, 0.010, 0.95),
            "trajectory_corrected_layer17_only": _metric(
                native, 0.015, 0.015, 0.94
            ),
            "frozen_uncorrected_composition": _metric(
                native, 0.050, 0.080, 0.85
            ),
            "trajectory_corrected_composition": _metric(
                native, 0.024, 0.040, 0.91
            ),
            "matched_double_deletion": _metric(
                native, 0.200, 0.200, 0.60
            ),
        },
        "resource_accounting": resources,
        "exact_resources_match_protocol": True,
        "latency_or_kernel_speed_claim": False,
    }


def _authority_metadata() -> dict[str, object]:
    binding = authority_module._public_protocol_binding(
        build_default_v8_layer17_family_lofo_protocol()
    )
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "receipt": {
            "receipt_sha256": "1" * 64,
            "receipt_file_sha256": "2" * 64,
        },
        "protocol": {
            key: binding[key]
            for key in (
                "protocol_artifact_sha256",
                "fit_membership_sha256",
                "family_alias_mapping_sha256",
                "fold_count",
                "folds",
            )
        },
        "corpus": {
            "corpus_artifact_sha256": binding["corpus_artifact_sha256"],
            "corpus_artifact_file_sha256": "3" * 64,
            "tokenizer_contract_sha256": binding["tokenizer_contract_sha256"],
            "fit_manifest_sha256": binding["fit_manifest_sha256"],
            "fit_role_file_sha256": binding["fit_role_file_sha256"],
            "example_count": 256,
            "block_count": 8,
            "examples_per_block": 32,
            "block_labels": V8_FAMILY_LOFO_FAMILY_ALIASES,
        },
        "access": dict(authority_module._AUTHORITY_ACCESS),
        "safety": dict(authority_module._PUBLIC_SAFETY),
    }
    result = {
        **payload,
        "authority_sha256": authority_module._domain_sha256(
            authority_module._AUTHORITY_DOMAIN,
            payload,
        ),
    }
    authority_module.validate_gemma3_layer17_family_lofo_authority_metadata(
        result
    )
    return result


def _materialization(authority: dict[str, object]) -> dict[str, object]:
    blocks = {
        alias: {
            "example_count": 32,
            "batch_count": 4,
            "logical_valid_tokens": 128,
            "supervised_tokens": 100,
        }
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
    }
    tokenization = {
        "block_count": 8,
        "block_labels": list(V8_FAMILY_LOFO_FAMILY_ALIASES),
        "example_count": 256,
        "examples_per_block": 32,
        "batch_count": 32,
        "logical_valid_tokens": 1024,
        "supervised_tokens": 800,
        "max_length": 128,
        "tokenization_batch_size": 8,
        "device": "cpu",
        "stream_catalog_sha256": "4" * 64,
        "blocks": blocks,
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "authority_sha256": authority["authority_sha256"],
        "tokenization": tokenization,
        "access": dict(authority_module._MATERIALIZATION_ACCESS),
        "safety": dict(authority_module._PUBLIC_SAFETY),
    }
    result = {
        **payload,
        "materialization_sha256": authority_module._domain_sha256(
            authority_module._MATERIALIZATION_DOMAIN,
            payload,
        ),
    }
    authority_module.validate_gemma3_layer17_family_lofo_materialization_metadata(
        result
    )
    return result


def _source_runtime_catalog(protocol: dict[str, object]) -> dict[str, object]:
    source = protocol["source_authority"]
    records = protocol["projection_contract"]["ordered_decoders"]
    layer10_order = (
        "gemma3.layer-10.cluster-28.modal-generator.same-layer-0.graph-node",
        "gemma3.layer-10.cluster-0.modal-generator.same-layer-1.graph-node",
        "gemma3.layer-10.cluster-34.modal-generator.same-layer-2.graph-node",
        "gemma3.layer-10.cluster-63.modal-generator.same-layer-3.graph-node",
    )
    layer17_order = tuple(record["node_name"] for record in records)
    payload: dict[str, object] = {
        "authenticated_bundle_file_sha256": source["composition_bundle"][
            "tensor_file_sha256"
        ],
        "frozen_primary_graph_sha256": source["composition_bundle"][
            "combined_primary_graph_sha256"
        ],
        "frozen_primary_traversal_order": (*layer10_order, *layer17_order),
        "layer10": {
            "graph_sha256": source["layer10"]["primary_graph_sha256"],
            "traversal_order": layer10_order,
            "lowering_sha256_by_node": dict(
                zip(
                    layer10_order,
                    (
                        "e07124c27cb4d61e5450c109a3a56d802da490ad04ca3056553ba34db822bc47",
                        "69c7dbb56ae0700eb2796ffe85ee9096658afb365742a5ae72fc54e024efa473",
                        "c2339759064eab6d6bbc0b79d22a76ba4d6aafc292d179585faa9aa8b4acf1f9",
                        "e23c892eca14ff5493e2f19c9e4960016a133677db89f2daccb13b7d087ba3f4",
                    ),
                    strict=True,
                )
            ),
        },
        "layer17": {
            "graph_sha256": source["layer17_decoder_source"][
                "source_edgeless_graph_sha256"
            ],
            "traversal_order": layer17_order,
            "lowering_sha256_by_node": dict(
                zip(
                    layer17_order,
                    (
                        "d69066c28e144b482787f2f66d7934e5a42b45b06dd80c66182c68df2eec5221",
                        "27752e63c5b796c0c903649d1b5ca5cf7326cb11971f0c25e490f7c97bd53ff5",
                        "501bf4f2faf924302db7322b56a91948bcfb158528e3657c718292984ccf7564",
                        "60a081e187a2dc50bea9f60f3c5fdbd3eb83e5dacf9a9d5cac16fc8292fb0fd3",
                    ),
                    strict=True,
                )
            ),
            "fragment_ids": tuple(record["fragment_id"] for record in records),
            "selection_sha256": (
                "98a916e2a5d90c246c333cbbf19838e0dcc5d86051541a23cc8430547993988e"
            ),
        },
    }
    result = {
        **payload,
        "catalog_sha256": lofo._domain_sha256(
            lofo._SOURCE_RUNTIME_CATALOG_DOMAIN,
            payload,
        ),
    }
    assert result["catalog_sha256"] == (
        lofo._EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256
    )
    return result


def _projection_metadata(protocol: dict[str, object]) -> dict[str, object]:
    projection = protocol["projection_contract"]
    records = projection["ordered_decoders"]
    return {
        "projection_method": (
            "float64_affine_sum_svd_pseudoinverse_minimum_norm"
        ),
        "node_order": [value["node_name"] for value in records],
        "basis_sha256_by_node": {
            value["node_name"]: value["computational_mode_basis_sha256"]
            for value in records
        },
        "mean_bias_sha256_by_node": {
            value["node_name"]: value["mean_bias_sha256"]
            for value in records
        },
        "decoder_basis_sha256_by_node": {
            value["node_name"]: value["decoder_basis_sha256"]
            for value in records
        },
        "affine_offset_sha256": projection["summed_mean_sha256"],
        "combined_basis_rank": 182,
        "observation_count": 100,
        "residual_width": 640,
        "target_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "coordinate_sha256_by_node": {
            value["node_name"]: "c" * 64 for value in records
        },
        "contribution_sha256_by_node": {
            value["node_name"]: "d" * 64 for value in records
        },
        "rmse": 0.1,
        "target_rms": 1.0,
        "nrmse": 0.1,
        "max_abs_error": 0.2,
        "offline_projection_only": True,
        "runtime_parameter_count": 0,
        "runtime_macs_per_token": 0,
    }


def _fold_report(
    protocol: dict[str, object],
    index: int,
    evaluation: dict[str, object],
    *,
    capture_sha256: str,
    source_runtime_catalog: dict[str, object],
) -> dict[str, object]:
    fold = protocol["folds"][index]
    projection = _projection_metadata(protocol)
    fit_projection = copy.deepcopy(projection)
    fit_projection["observation_count"] = 700
    held_projection = copy.deepcopy(projection)
    held_projection["observation_count"] = 100
    receipt: dict[str, object] = {
        "capture_sha256": capture_sha256,
        "protocol_fold_sha256": fold["artifact_sha256"],
        "authenticated_stream_sha256": "4" * 64,
        "fit_family_aliases": tuple(fold["training_family_aliases"]),
        "held_family_alias": fold["held_family_alias"],
        "fit_row_key_sha256": f"{index + 1:064x}",
        "held_row_key_sha256": f"{index + 101:064x}",
        "fit_observations": 700,
        "held_observations": 100,
        "fit_sequences": 224,
        "held_sequences": 32,
        "held_family_excluded_from_projection_fit_and_generator_fit": True,
        "held_fisher_weights_preserved_without_held_family_normalization": True,
        "held_projection_used_only_as_fixed_basis_generator_evaluation_target": True,
        "fit_projection": fit_projection,
        "held_projection": held_projection,
    }
    receipt["fit_split_sha256"] = lofo._reported_split_sha256(
        receipt,
        protocol_sha256=protocol["artifact_sha256"],
        role="fit",
    )
    receipt["held_split_sha256"] = lofo._reported_split_sha256(
        receipt,
        protocol_sha256=protocol["artifact_sha256"],
        role="held",
    )
    node_order = [
        value["node_name"]
        for value in protocol["projection_contract"]["ordered_decoders"]
    ]
    lowering_hashes = {
        name: f"{index + node_index + 501:064x}"
        for node_index, name in enumerate(node_order)
    }
    generator_hashes = {
        name: f"{index + node_index + 601:064x}"
        for node_index, name in enumerate(node_order)
    }
    decoder_hashes = {
        value["node_name"]: value["decoder_basis_sha256"]
        for value in protocol["projection_contract"]["ordered_decoders"]
    }
    mean_hashes = {
        value["node_name"]: value["mean_bias_sha256"]
        for value in protocol["projection_contract"]["ordered_decoders"]
    }
    correction_fit = {
        "graph_sha256": f"{index + 201:064x}",
        "node_order": node_order,
        "parameter_count": 163_094,
        "macs_per_token": 160_352,
        "interaction_count": 0,
        "source_decoder_basis_sha256_by_node": decoder_hashes,
        "source_mean_bias_sha256_by_node": mean_hashes,
        "lowering_sha256_by_node": lowering_hashes,
        "generator_plan_sha256_by_node": generator_hashes,
    }
    corrected_layer17_graph_sha256 = f"{index + 201:064x}"
    corrected_composition_graph_sha256 = f"{index + 401:064x}"
    composition_receipt = lofo._build_corrected_composition_receipt(
        source_runtime_catalog=source_runtime_catalog,
        corrected_layer17_graph_sha256=corrected_layer17_graph_sha256,
        corrected_layer17_lowering_sha256_by_node=lowering_hashes,
        corrected_composition_graph_sha256=(
            corrected_composition_graph_sha256
        ),
    )
    return {
        "fold_index": fold["fold_index"],
        "fold_id": fold["fold_id"],
        "held_family_alias": fold["held_family_alias"],
        "training_family_aliases": fold["training_family_aliases"],
        "protocol_fold_sha256": fold["artifact_sha256"],
        "row_receipt": receipt,
        "correction_fit": correction_fit,
        "corrected_layer17_graph_sha256": corrected_layer17_graph_sha256,
        "corrected_composition_graph_sha256": (
            corrected_composition_graph_sha256
        ),
        "corrected_lowering_sha256_by_node": lowering_hashes,
        "composition_receipt": composition_receipt,
        "evaluation": evaluation,
    }


def _report() -> dict[str, object]:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    source = protocol["source_authority"]
    model = source["model"]
    composition = source["composition_bundle"]
    layer10 = source["layer10"]
    layer17 = source["layer17_decoder_source"]
    projection = protocol["projection_contract"]
    authority = _authority_metadata()
    materialization = _materialization(authority)
    source_runtime_catalog = _source_runtime_catalog(protocol)
    source_layer10 = source_runtime_catalog["layer10"]
    source_layer17 = source_runtime_catalog["layer17"]
    capture_payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_layer17_native_layer10_trajectory_rows"
        ),
        "format_version": 2,
        "scientific_role": "paired_native_and_layer10_compiled_layer17_rows",
        "source_safe": True,
        "contains_tensors": False,
        "contains_prompt_text": False,
        "contains_prompt_identities": False,
        "contains_token_ids": False,
        "condition": "generated",
        "affected_layer_ordinals": [10],
        "alignment": {
            "fragment_count": 4,
            "sequences": 256,
            "observations": 800,
            "row_key_sha256": "8" * 64,
        },
        "model_fingerprint": model["adapter_model_fingerprint"],
        "layer10": {
            "graph_sha256": source_layer10["graph_sha256"],
            "traversal_order": source_layer10["traversal_order"],
            "ordered_lowering_sha256s": tuple(
                source_layer10["lowering_sha256_by_node"][name]
                for name in source_layer10["traversal_order"]
            ),
        },
        "layer17": {
            "layer_ordinal": 17,
            "selection_sha256": source_layer17["selection_sha256"],
            "leaf_activation_site": "layer.17.mlp.down_input",
            "fragment_ids": source_layer17["fragment_ids"],
            "teacher_role": (
                "native_layer17_fragment_residual_contribution_on_each_trajectory"
            ),
            "input_roles": {
                "native": "native_model_layer17_normalized_mlp_input",
                "compiled": (
                    "layer17_normalized_mlp_input_after_frozen_"
                    "layer10_generated_graph"
                ),
            },
            "full_output_roles": {
                "native": "native_model_layer17_full_mlp_output",
                "compiled": (
                    "native_layer17_full_mlp_output_after_frozen_"
                    "layer10_generated_graph"
                ),
            },
            "compact_retained_output_role": (
                "authenticated_layer17_compact_mlp_output_on_"
                "layer10_compiled_input"
            ),
            "algebraic_compact_retained_audit_role": (
                "compiled_full_minus_selected_compiled_contributions_"
                "numerical_audit_only"
            ),
            "a3_target_role": (
                "native_full_minus_exact_authenticated_compact_retained_output"
            ),
            "compact_executor": {
                "graph_sha256": source_layer17["graph_sha256"],
                "traversal_order": source_layer17["traversal_order"],
                "ordered_lowering_sha256s": tuple(
                    source_layer17["lowering_sha256_by_node"][name]
                    for name in source_layer17["traversal_order"]
                ),
                "interaction_count": 0,
                "affected_layer_ordinals": [17],
            },
        },
        "compact_retained_numerical_audit": {
            "role": (
                "compiled_full_minus_selected_compiled_contributions_"
                "numerical_audit_only"
            ),
            "difference_definition": (
                "exact_compact_retained_minus_compiled_full_minus_"
                "selected_compiled_contributions"
            ),
            "rms_difference": 1e-6,
            "max_abs_difference": 1e-5,
        },
    }
    capture = {
        **capture_payload,
        "capture_sha256": hashlib.sha256(
            lofo._CAPTURE_METADATA_DOMAIN
            + lofo._canonical_json_bytes(capture_payload)
        ).hexdigest(),
    }
    evaluations = [_evaluation(index) for index in range(8)]
    folds = [
        _fold_report(
            protocol,
            index,
            evaluations[index],
            capture_sha256=capture["capture_sha256"],
            source_runtime_catalog=source_runtime_catalog,
        )
        for index in range(8)
    ]
    aggregate = lofo.aggregate_trajectory_correction_lofo_folds(evaluations)
    decision = lofo.evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=evaluations,
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=True,
        compact_replay_algebraic_equivalence_audit=True,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )
    authorization = {
        "authorization_kind": "frozen_composition_then_fit_only_family_lofo",
        "authorization_completed_before_fit_open": True,
        "protocol_sha256": protocol["artifact_sha256"],
        "bundle": {
            "bundle_file_sha256": composition["tensor_file_sha256"],
            "composition_payload_sha256": composition[
                "composition_payload_sha256"
            ],
            "combined_edgeless_graph_sha256": composition[
                "combined_edgeless_graph_sha256"
            ],
            "combined_primary_graph_sha256": composition[
                "combined_primary_graph_sha256"
            ],
            "model_fingerprint": model["adapter_model_fingerprint"],
            "layer10_candidate_tensor_file_sha256": layer10[
                "candidate_tensor_file_sha256"
            ],
            "layer10_candidate_scientific_payload_sha256": layer10[
                "candidate_scientific_payload_sha256"
            ],
            "layer17_candidate_tensor_file_sha256": layer17[
                "candidate_tensor_file_sha256"
            ],
            "layer17_candidate_scientific_payload_sha256": layer17[
                "candidate_scientific_payload_sha256"
            ],
            "layer17_edgeless_graph_sha256": layer17[
                "source_edgeless_graph_sha256"
            ],
        },
        "source_runtime_catalog": source_runtime_catalog,
        "fit_authority": authority,
        "fit_authority_sha256": authority["authority_sha256"],
        "fit_opened": True,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "source_safe": True,
    }
    return lofo.build_trajectory_correction_lofo_report(
        protocol=protocol,
        authorization=authorization,
        runtime={
            "model_id": model["model_id"],
            "requested_revision": model["requested_revision"],
            "model_fingerprint": model["adapter_model_fingerprint"],
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
            "vocabulary_chunk_size": 16384,
            "randomness": {
                **lofo._RANDOMNESS_RECIPE,
                "recipe_sha256": lofo._domain_sha256(
                    lofo._RANDOMNESS_RECIPE_DOMAIN,
                    lofo._RANDOMNESS_RECIPE,
                ),
                "torch_manual_seed_applied": True,
                "torch_cuda_manual_seed_all_applied": False,
            },
        },
        fit_collection={
            "materialization": materialization,
            "authenticated_stream_sha256": "4" * 64,
            "capture": capture,
            "capture_count": 1,
            "captured_examples": 256,
            "captured_sequences": 256,
            "captured_observations": 800,
            "model_rows_recollected_per_fold": False,
            "compiled_keep_replay_audit": {
                "construction": protocol["target_contract"]["compiled_keep_pass"],
                "capture": protocol["target_contract"]["compiled_keep_capture"],
                "runtime_deletion_operator_equivalent": True,
                "layer17_compact_graph_sha256": layer17[
                    "source_edgeless_graph_sha256"
                ],
                "ordered_layer17_compact_lowering_sha256s": tuple(
                    source_layer17["lowering_sha256_by_node"][name]
                    for name in source_layer17["traversal_order"]
                ),
                "exact_compact_replay_used_as_target": True,
                "algebraic_reference_used_as_target": False,
                "algebraic_equivalence_rmse": 1e-6,
                "algebraic_equivalence_max_abs_difference": 1e-5,
                "maximum_algebraic_equivalence_rmse": protocol[
                    "target_contract"
                ]["maximum_algebraic_equivalence_rmse"],
                "maximum_algebraic_equivalence_max_abs_difference": protocol[
                    "target_contract"
                ]["maximum_algebraic_equivalence_max_abs_difference"],
            },
            "a3_target_construction": protocol["target_contract"][
                "raw_target_formula"
            ],
            "decoder_span_sha256": projection["decoder_span_sha256"],
            "summed_mean_sha256": projection["summed_mean_sha256"],
        },
        folds=folds,
        aggregate=aggregate,
        resources=lofo._EXPECTED_EXACT_RESOURCES,
        decision=decision,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )


def test_aggregate_and_frozen_gate_table_pass_on_eight_good_families() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    evaluations = [_evaluation(index) for index in range(8)]
    aggregate = lofo.aggregate_trajectory_correction_lofo_folds(evaluations)
    decision = lofo.evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=evaluations,
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=True,
        compact_replay_algebraic_equivalence_audit=True,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )

    assert aggregate["family_count"] == 8
    assert decision["all_required_gates_pass"] is True
    assert decision["derived_metrics"][
        "family_macro_interaction_excess_nll"
    ] == pytest.approx(-0.001)
    assert len(decision["gate_table"]) == 19


def test_gates_fail_closed_on_insufficient_family_kl_improvements() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    evaluations = [_evaluation(index) for index in range(8)]
    for value in evaluations[5:]:
        native = value["native"]["nll_per_token"]
        value["conditions"]["trajectory_corrected_composition"] = _metric(
            native, 0.024, 0.081, 0.91
        )
    aggregate = lofo.aggregate_trajectory_correction_lofo_folds(evaluations)
    decision = lofo.evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=evaluations,
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=True,
        compact_replay_algebraic_equivalence_audit=True,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )
    gates = {row["gate_id"]: row for row in decision["gate_table"]}

    assert gates["held_family_kl_improvement_count"]["observed"] == 5
    assert gates["held_family_kl_improvement_count"]["passed"] is False
    assert decision["all_required_gates_pass"] is False


def test_gates_fail_closed_on_invalid_deletion_denominator_and_audit() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    evaluations = [_evaluation(index) for index in range(8)]
    for value in evaluations:
        native = value["native"]["nll_per_token"]
        value["conditions"]["matched_double_deletion"] = _metric(
            native, 0.0, 0.0, 1.0
        )
    aggregate = lofo.aggregate_trajectory_correction_lofo_folds(evaluations)
    decision = lofo.evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=evaluations,
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=True,
        compact_replay_algebraic_equivalence_audit=False,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )
    gates = {row["gate_id"]: row for row in decision["gate_table"]}

    assert decision["derived_metrics"][
        "deletion_nll_recovery_denominator_valid"
    ] is False
    assert gates["compact_replay_algebraic_equivalence_audit"]["passed"] is False
    assert decision["all_required_gates_pass"] is False


def _synthetic_trajectory_capture() -> tuple[object, dict[str, str]]:
    aliases = V8_FAMILY_LOFO_FAMILY_ALIASES
    capture_order = (
        aliases[3],
        aliases[0],
        aliases[7],
        aliases[1],
        aliases[6],
        aliases[2],
        aliases[5],
        aliases[4],
    )
    row_keys = tuple(
        (f"opaque-{alias}", position)
        for position in (0, 1)
        for alias in capture_order
    )
    family_alias_by_example = {
        f"opaque-{alias}": alias for alias in aliases
    }
    observations = len(row_keys)
    width = 3
    fragment_ids = tuple(f"fragment-{index}" for index in range(4))
    compiled_input = torch.arange(
        observations * width,
        dtype=torch.float64,
    ).reshape(observations, width)
    native_input = compiled_input + 1_000.0
    native_rows = AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=native_input,
                contributions=torch.full(
                    (observations, width),
                    float(index + 1),
                    dtype=torch.float64,
                ),
                fisher_weights=(
                    torch.arange(1, observations + 1, dtype=torch.float64)
                    + index * 0.25
                ),
                sequences=len(aliases),
            )
            for index, fragment_id in enumerate(fragment_ids)
        },
        row_keys=row_keys,
    )
    compiled_rows = AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=compiled_input,
                contributions=torch.full(
                    (observations, width),
                    float(index + 5),
                    dtype=torch.float64,
                ),
                fisher_weights=torch.ones(observations, dtype=torch.float64),
                sequences=len(aliases),
            )
            for index, fragment_id in enumerate(fragment_ids)
        },
        row_keys=row_keys,
    )
    compact = torch.full(
        (observations, width),
        7.0,
        dtype=torch.float64,
    )
    exact_target = torch.arange(
        observations * width,
        dtype=torch.float64,
    ).reshape(observations, width) / 10.0
    row_pair = lofo.Gemma3Layer17TrajectoryRowPair(
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full_mlp_output=compact + exact_target,
        compiled_full_mlp_output=torch.zeros_like(compact),
        compiled_compact_retained_mlp_output=compact,
        model_fingerprint="a" * 64,
        layer17_selection_sha256="b" * 64,
        layer17_leaf_activation_site="layer.17.mlp.input",
        fragment_ids=fragment_ids,
        layer10_graph_sha256="c" * 64,
        layer10_traversal_order=tuple(f"layer10-node-{i}" for i in range(4)),
        ordered_layer10_lowering_sha256s=tuple(
            f"{index + 1:064x}" for index in range(4)
        ),
        layer17_compact_graph_sha256="d" * 64,
        layer17_compact_traversal_order=tuple(
            f"layer17-node-{i}" for i in range(4)
        ),
        ordered_layer17_compact_lowering_sha256s=tuple(
            f"{index + 5:064x}" for index in range(4)
        ),
    )
    return row_pair, family_alias_by_example


def _synthetic_source_graph_and_lowerings(
    fragment_ids: tuple[str, ...],
) -> tuple[object, dict[str, object]]:
    node_names = tuple(f"node-{index}" for index in range(4))
    lowerings = {}
    for index, (name, fragment_id) in enumerate(
        zip(node_names, fragment_ids, strict=True)
    ):
        fragment_sha256 = f"{index + 17:064x}"
        lowerings[name] = SimpleNamespace(
            selected_fragment_sha256=fragment_sha256,
            fragment_plan=SimpleNamespace(
                fragments=(
                    SimpleNamespace(
                        fragment_id=fragment_id,
                        artifact_sha256=fragment_sha256,
                    ),
                )
            ),
            computational_mode_basis=SimpleNamespace(name=name),
        )
    graph = SimpleNamespace(
        traversal_order=node_names,
        validate_integrity=lambda: None,
    )
    return graph, lowerings


def test_lean_fit_view_retains_only_exact_fold_inputs_and_validates_lineage() -> None:
    row_pair, _ = _synthetic_trajectory_capture()
    view = lofo._build_trajectory_correction_fit_view(row_pair)

    assert tuple(field.name for field in fields(view)) == (
        "compiled_input",
        "a3_target",
        "native_fisher_weights_by_fragment",
        "row_keys",
        "row_key_sha256",
        "sequences",
        "fragment_ids",
        "capture_sha256",
    )
    first_fragment = row_pair.fragment_ids[0]
    assert view.compiled_input.data_ptr() == row_pair.compiled_rows.rows_by_fragment[
        first_fragment
    ].inputs.data_ptr()
    assert torch.equal(
        view.a3_target,
        lofo.build_a3_raw_mlp_target(
            row_pair.native_full_mlp_output,
            row_pair.compiled_compact_retained_mlp_output,
        ),
    )
    assert not torch.equal(
        view.a3_target,
        row_pair.native_full_mlp_output
        - row_pair.algebraic_compact_retained_mlp_output,
    )
    assert all(
        view.native_fisher_weights_by_fragment[fragment_id].data_ptr()
        == row_pair.native_rows.rows_by_fragment[
            fragment_id
        ].fisher_weights.data_ptr()
        for fragment_id in row_pair.fragment_ids
    )
    with pytest.raises(ValueError, match="row-key lineage"):
        replace(view, row_key_sha256="0" * 64)


def test_direct_fit_view_builder_preserves_order_and_native_fisher(
    monkeypatch,
) -> None:
    row_pair, family_alias_by_example = _synthetic_trajectory_capture()
    view = lofo._build_trajectory_correction_fit_view(row_pair)
    graph, lowerings = _synthetic_source_graph_and_lowerings(
        row_pair.fragment_ids
    )
    aliases = V8_FAMILY_LOFO_FAMILY_ALIASES
    held = aliases[7]
    training = (
        aliases[3],
        aliases[0],
        aliases[6],
        aliases[1],
        aliases[5],
        aliases[2],
        aliases[4],
    )
    calls: list[dict[str, object]] = []

    class Projection:
        def __init__(self, observations: int) -> None:
            self.observations = observations

        def metadata(self) -> dict[str, object]:
            return {"observation_count": self.observations}

    def fake_projection(**kwargs):
        calls.append(kwargs)
        target = kwargs["target"]
        rows = AlignedFragmentRows(
            rows_by_fragment={
                kwargs["fragment_id_by_node"][name]: LayerFragmentRows(
                    inputs=kwargs["inputs"],
                    contributions=torch.zeros_like(target),
                    fisher_weights=kwargs["fisher_weights_by_node"][name],
                    sequences=kwargs["sequences"],
                )
                for name in kwargs["node_order"]
            },
            row_keys=kwargs["row_keys"],
        )
        return rows, Projection(target.shape[0])

    def forbidden_legacy_path(*_args, **_kwargs):
        raise AssertionError("legacy partition/concatenation path was called")

    monkeypatch.setattr(
        lofo,
        "build_projected_correction_rows",
        fake_projection,
    )
    monkeypatch.setattr(
        lofo,
        "partition_aligned_fragment_rows_by_family",
        forbidden_legacy_path,
        raising=False,
    )
    monkeypatch.setattr(
        lofo,
        "_concatenate_family_rows",
        forbidden_legacy_path,
        raising=False,
    )
    capture_fisher = {
        fragment_id: weights.clone()
        for fragment_id, weights in view.native_fisher_weights_by_fragment.items()
    }

    fit_rows, held_rows, receipt = (
        lofo._build_trajectory_correction_fold_rows_from_fit_view(
            view,
            family_alias_by_example=family_alias_by_example,
            training_family_aliases=training,
            held_family_alias=held,
            fold_sha256="e" * 64,
            protocol_sha256="f" * 64,
            authenticated_stream_sha256="1" * 64,
            source_graph=graph,
            source_lowerings_by_node=lowerings,
        )
    )

    family_indices = {
        alias: torch.tensor(
            [
                index
                for index, (example_id, _) in enumerate(view.row_keys)
                if family_alias_by_example[example_id] == alias
            ],
            dtype=torch.long,
        )
        for alias in aliases
    }
    fit_indices = torch.cat(tuple(family_indices[alias] for alias in training))
    held_indices = family_indices[held]
    assert len(calls) == 2
    assert calls[0]["row_keys"] == tuple(
        view.row_keys[int(index)] for index in fit_indices.tolist()
    )
    assert calls[1]["row_keys"] == tuple(
        view.row_keys[int(index)] for index in held_indices.tolist()
    )
    assert torch.equal(
        calls[0]["inputs"],
        view.compiled_input.index_select(0, fit_indices),
    )
    assert torch.equal(
        calls[0]["target"],
        view.a3_target.index_select(0, fit_indices),
    )
    assert torch.equal(
        calls[1]["inputs"],
        view.compiled_input.index_select(0, held_indices),
    )
    assert torch.equal(
        calls[1]["target"],
        view.a3_target.index_select(0, held_indices),
    )
    for name, fragment_id in zip(
        graph.traversal_order,
        view.fragment_ids,
        strict=True,
    ):
        raw = capture_fisher[fragment_id]
        expected_fit = torch.cat(
            tuple(
                (
                    raw.index_select(0, family_indices[alias])
                    / raw.index_select(0, family_indices[alias]).sum()
                    / 7
                )
                for alias in training
            )
        )
        assert torch.equal(
            calls[0]["fisher_weights_by_node"][name],
            expected_fit,
        )
        assert torch.equal(
            calls[1]["fisher_weights_by_node"][name],
            raw.index_select(0, held_indices),
        )
        assert torch.equal(
            view.native_fisher_weights_by_fragment[fragment_id],
            capture_fisher[fragment_id],
        )
    assert fit_rows.sequences == 7
    assert held_rows.sequences == 1
    assert receipt["fit_family_aliases"] == list(training)
    assert receipt["held_family_alias"] == held
    assert receipt["fit_projection"]["observation_count"] == len(fit_indices)
    assert receipt["held_projection"]["observation_count"] == len(held_indices)


def test_runner_releases_capture_before_direct_fold_building() -> None:
    runner_source = inspect.getsource(
        lofo.run_gemma3_l10_l17_trajectory_correction_lofo
    )
    direct_builder_source = inspect.getsource(
        lofo._build_trajectory_correction_fold_rows_from_fit_view
    )

    assert runner_source.index("capture_metadata = row_pair.metadata()") < (
        runner_source.index("fit_view = _build_trajectory_correction_fit_view")
    )
    assert runner_source.index("del row_pair") < runner_source.index(
        "for index, fold in enumerate"
    )
    assert "_build_trajectory_correction_fold_rows_from_fit_view(" in runner_source
    assert "partition_aligned_fragment_rows_by_family" not in direct_builder_source
    assert "_concatenate_family_rows" not in direct_builder_source


def test_authorization_authenticates_bundle_before_opening_fit(monkeypatch) -> None:
    events: list[str] = []
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    fit = protocol["source_authority"]["calibration_a_fit"]
    authority_safe = {
        "corpus": {
            "fit_manifest_sha256": fit["fit_manifest_sha256"],
            "fit_role_file_sha256": fit["fit_source_file_sha256"],
        }
    }
    fake_bundle = SimpleNamespace(binding={})
    fake_authority = object()

    monkeypatch.setattr(
        lofo,
        "_bundle_authority",
        lambda _path: events.append("bundle") or fake_bundle,
    )
    monkeypatch.setattr(lofo, "_validate_bundle_against_protocol", lambda *_: None)
    monkeypatch.setattr(
        lofo,
        "load_gemma3_layer17_family_lofo_authority",
        lambda **_kwargs: events.append("fit") or fake_authority,
    )
    monkeypatch.setattr(
        lofo,
        "_authority_metadata",
        lambda _authority: ("a" * 64, authority_safe),
    )
    monkeypatch.setattr(
        lofo,
        "validate_gemma3_layer17_family_lofo_authority_metadata",
        lambda _value: None,
    )
    monkeypatch.setattr(
        lofo,
        "_load_authenticated_protocol",
        lambda *_args, **_kwargs: {
            "artifact_sha256": FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        },
    )
    monkeypatch.setattr(lofo, "_reject_forbidden_output_fields", lambda *_args, **_kwargs: None)

    lofo._authenticate_before_fit_access(
        bundle_path="bundle.pt",
        corpus_receipt_path="receipt.json",
        corpus_artifact_path="corpus.json",
        fit_input_path="fit.json",
    )

    assert events == ["bundle", "fit"]


def test_report_validator_replays_metrics_and_rejects_source_leakage(tmp_path: Path) -> None:
    report = _report()
    validated = lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
        report
    )
    assert validated["decision"]["all_required_gates_pass"] is True

    tampered = copy.deepcopy(report)
    tampered["decision"]["all_required_gates_pass"] = False
    payload = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = lofo._domain_sha256(lofo._REPORT_DOMAIN, payload)
    with pytest.raises(ValueError, match="decision"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(tampered)

    leaked = copy.deepcopy(report)
    leaked["fit_collection"]["prompt"] = "do not serialize me"
    payload = {key: value for key, value in leaked.items() if key != "report_sha256"}
    leaked["report_sha256"] = lofo._domain_sha256(lofo._REPORT_DOMAIN, payload)
    with pytest.raises(ValueError, match="forbidden"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(leaked)

    output = tmp_path / "report.json"
    lofo.save_gemma3_l10_l17_trajectory_correction_lofo_report(output, report)
    with pytest.raises(FileExistsError, match="overwrite"):
        lofo.save_gemma3_l10_l17_trajectory_correction_lofo_report(output, report)


def _rehash(report: dict[str, object]) -> None:
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = lofo._domain_sha256(lofo._REPORT_DOMAIN, payload)


def _rehash_capture_and_splits(report: dict[str, object]) -> None:
    capture = report["fit_collection"]["capture"]
    capture["capture_sha256"] = lofo._capture_metadata_sha256(capture)
    for fold in report["folds"]:
        receipt = fold["row_receipt"]
        receipt["capture_sha256"] = capture["capture_sha256"]
        receipt["fit_split_sha256"] = lofo._reported_split_sha256(
            receipt,
            protocol_sha256=report["protocol"]["artifact_sha256"],
            role="fit",
        )
        receipt["held_split_sha256"] = lofo._reported_split_sha256(
            receipt,
            protocol_sha256=report["protocol"]["artifact_sha256"],
            role="held",
        )
    _rehash(report)


def test_report_rejects_old_linear_projection_and_wrong_affine_mean() -> None:
    old_linear = _report()
    old_linear["folds"][0]["row_receipt"]["fit_projection"][
        "projection_method"
    ] = "float64_svd_pseudoinverse_minimum_norm"
    _rehash(old_linear)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            old_linear
        )

    wrong_mean = _report()
    wrong_mean["folds"][0]["row_receipt"]["held_projection"][
        "affine_offset_sha256"
    ] = "0" * 64
    _rehash(wrong_mean)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_mean
        )


def test_report_rejects_unrelated_fold_generator_and_lowering_artifacts() -> None:
    wrong_generator = _report()
    generator_map = wrong_generator["folds"][0]["correction_fit"][
        "generator_plan_sha256_by_node"
    ]
    removed = next(iter(generator_map))
    del generator_map[removed]
    _rehash(wrong_generator)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_generator
        )

    wrong_lowering = _report()
    lowering_map = wrong_lowering["folds"][0][
        "corrected_lowering_sha256_by_node"
    ]
    name = next(iter(lowering_map))
    lowering_map[name] = "0" * 64
    _rehash(wrong_lowering)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_lowering
        )


def test_report_rejects_randomness_count_and_capture_safety_tampering() -> None:
    wrong_seed = _report()
    wrong_seed["runtime"]["randomness"]["torch_seed"] += 1
    _rehash(wrong_seed)
    with pytest.raises(ValueError, match="randomness"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_seed
        )

    wrong_count = _report()
    wrong_count["fit_collection"]["capture_count"] = 2
    _rehash(wrong_count)
    with pytest.raises(ValueError, match="capture/compiled-keep"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_count
        )

    unsafe_capture = _report()
    capture = unsafe_capture["fit_collection"]["capture"]
    capture["contains_tensors"] = True
    capture["capture_sha256"] = lofo._capture_metadata_sha256(capture)
    for fold in unsafe_capture["folds"]:
        fold["row_receipt"]["capture_sha256"] = capture["capture_sha256"]
        fold["row_receipt"]["fit_split_sha256"] = lofo._reported_split_sha256(
            fold["row_receipt"],
            protocol_sha256=unsafe_capture["protocol"]["artifact_sha256"],
            role="fit",
        )
        fold["row_receipt"]["held_split_sha256"] = lofo._reported_split_sha256(
            fold["row_receipt"],
            protocol_sha256=unsafe_capture["protocol"]["artifact_sha256"],
            role="held",
        )
    _rehash(unsafe_capture)
    with pytest.raises(ValueError, match="capture executable"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            unsafe_capture
        )


def test_report_rejects_fold_sequence_and_projection_count_tampering() -> None:
    wrong_sequences = _report()
    wrong_sequences["folds"][0]["row_receipt"]["fit_sequences"] = 223
    _rehash(wrong_sequences)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_sequences
        )

    wrong_projection_count = _report()
    wrong_projection_count["folds"][0]["row_receipt"]["fit_projection"][
        "observation_count"
    ] = 699
    _rehash(wrong_projection_count)
    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_projection_count
        )


def test_report_rejects_fold_observation_coverage_tampering() -> None:
    wrong_complement = _report()
    receipt = wrong_complement["folds"][0]["row_receipt"]
    receipt["fit_observations"] = 701
    receipt["fit_projection"]["observation_count"] = 701
    receipt["fit_split_sha256"] = lofo._reported_split_sha256(
        receipt,
        protocol_sha256=wrong_complement["protocol"]["artifact_sha256"],
        role="fit",
    )
    _rehash(wrong_complement)
    with pytest.raises(ValueError, match="observation coverage"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_complement
        )

    wrong_held_sum = _report()
    receipt = wrong_held_sum["folds"][0]["row_receipt"]
    receipt["fit_observations"] = 699
    receipt["held_observations"] = 101
    receipt["fit_projection"]["observation_count"] = 699
    receipt["held_projection"]["observation_count"] = 101
    receipt["fit_split_sha256"] = lofo._reported_split_sha256(
        receipt,
        protocol_sha256=wrong_held_sum["protocol"]["artifact_sha256"],
        role="fit",
    )
    receipt["held_split_sha256"] = lofo._reported_split_sha256(
        receipt,
        protocol_sha256=wrong_held_sum["protocol"]["artifact_sha256"],
        role="held",
    )
    _rehash(wrong_held_sum)
    with pytest.raises(ValueError, match="observation coverage"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_held_sum
        )


def test_a3_fold_builder_has_no_arbitrary_compiled_keep_input() -> None:
    import inspect

    parameters = inspect.signature(
        lofo.build_trajectory_correction_fold_rows
    ).parameters
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()

    assert "compiled_keep_output" not in parameters
    assert protocol["target_contract"]["raw_target_formula"] == (
        "native_full_layer17_mlp_operator_output-"
        "compiled_keep_layer17_mlp_operator_output"
    )
    native = torch.tensor([[3.0, -1.0]], dtype=torch.float64)
    compact = torch.tensor([[1.5, -2.0]], dtype=torch.float64)
    assert torch.equal(
        lofo.build_a3_raw_mlp_target(native, compact),
        torch.tensor([[1.5, 1.0]], dtype=torch.float64),
    )


def test_report_rejects_rehashed_source_catalog_and_capture_catalog_drift() -> None:
    report = _report()
    catalog = report["authorization"]["source_runtime_catalog"]
    assert catalog["catalog_sha256"] == (
        lofo._EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256
    )

    node = catalog["layer10"]["traversal_order"][0]
    catalog["layer10"]["lowering_sha256_by_node"][node] = "0" * 64
    report["fit_collection"]["capture"]["layer10"][
        "ordered_lowering_sha256s"
    ][0] = "0" * 64
    catalog_payload = {
        key: value for key, value in catalog.items() if key != "catalog_sha256"
    }
    catalog["catalog_sha256"] = lofo._domain_sha256(
        lofo._SOURCE_RUNTIME_CATALOG_DOMAIN,
        catalog_payload,
    )
    _rehash_capture_and_splits(report)

    with pytest.raises(ValueError, match="runtime catalog hash"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(report)


@pytest.mark.parametrize(
    ("capture_path", "replacement"),
    (
        (("layer17", "selection_sha256"), "0" * 64),
        (("layer17", "fragment_ids", 0), "wrong-fragment"),
        (("layer10", "ordered_lowering_sha256s", 0), "0" * 64),
        (
            (
                "layer17",
                "compact_executor",
                "ordered_lowering_sha256s",
                0,
            ),
            "0" * 64,
        ),
    ),
)
def test_capture_catalog_must_match_pinned_runtime_catalog(
    capture_path: tuple[object, ...],
    replacement: str,
) -> None:
    report = _report()
    target: object = report["fit_collection"]["capture"]
    for key in capture_path[:-1]:
        target = target[key]
    target[capture_path[-1]] = replacement
    _rehash_capture_and_splits(report)

    with pytest.raises(ValueError, match="capture executable"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(report)


def test_report_binds_corrected_composition_graph_and_receipt() -> None:
    wrong_graph = _report()
    wrong_graph["folds"][0]["corrected_composition_graph_sha256"] = "0" * 64
    _rehash(wrong_graph)
    with pytest.raises(ValueError, match="corrected composition lineage"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_graph
        )

    wrong_receipt = _report()
    wrong_receipt["folds"][0]["composition_receipt"]["replacement_policy"][
        "interactions_preserved_exactly"
    ] = False
    receipt = wrong_receipt["folds"][0]["composition_receipt"]
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = lofo._domain_sha256(
        lofo._COMPOSITION_RECEIPT_DOMAIN,
        receipt_payload,
    )
    _rehash(wrong_receipt)
    with pytest.raises(ValueError, match="corrected composition lineage"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(
            wrong_receipt
        )


@pytest.mark.parametrize(
    ("mutation",),
    (
        ("extra_field",),
        ("invalid_target_sha256",),
        ("missing_coordinate_node",),
        ("negative_rmse",),
        ("inconsistent_nrmse",),
    ),
)
def test_projection_metadata_is_exact_and_diagnostics_are_consistent(
    mutation: str,
) -> None:
    report = _report()
    projection = report["folds"][0]["row_receipt"]["fit_projection"]
    if mutation == "extra_field":
        projection["uncommitted_diagnostic"] = 0.0
    elif mutation == "invalid_target_sha256":
        projection["target_sha256"] = "not-a-sha"
    elif mutation == "missing_coordinate_node":
        del projection["coordinate_sha256_by_node"][
            next(iter(projection["coordinate_sha256_by_node"]))
        ]
    elif mutation == "negative_rmse":
        projection["rmse"] = -0.1
    elif mutation == "inconsistent_nrmse":
        projection["nrmse"] = 0.2
    else:  # pragma: no cover - parametrization is frozen above.
        raise AssertionError(mutation)
    _rehash(report)

    with pytest.raises(ValueError, match="projection/refit"):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(report)


@pytest.mark.parametrize(
    ("record", "extra_field", "match"),
    (
        ("authorization", "validation_was_opened", "authorization fields"),
        ("runtime", "experimental_runtime_mode", "runtime fields"),
        ("fit_collection", "held_roles_were_opened", "fit collection fields"),
        ("row_receipt", "held_used_for_generator_fit", "projection/refit"),
        ("correction_fit", "decoder_basis_was_refit", "projection/refit"),
    ),
)
def test_report_rejects_extra_nested_claim_fields(
    record: str,
    extra_field: str,
    match: str,
) -> None:
    report = _report()
    if record in {"authorization", "runtime", "fit_collection"}:
        target = report[record]
    else:
        target = report["folds"][0][record]
    target[extra_field] = True
    _rehash(report)

    with pytest.raises(ValueError, match=match):
        lofo.validate_gemma3_l10_l17_trajectory_correction_lofo_report(report)
