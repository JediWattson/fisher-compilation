from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import stat

import pytest
import torch

import fisher_graph.gemma3_a5_same_shape_locality as locality_module
import fisher_graph.gemma3_l10_l17_a5c_report as report
import fisher_graph.gemma3_l10_l17_a5c_prepublication_bundle as prepublication
import fisher_graph.gemma3_l10_l17_a5c_family_ridge_cv as cv_module
import fisher_graph.gemma3_l10_l17_trajectory_correction_lofo as scorer_module
from fisher_graph.gemma3_l10_l17_a5c_breadth_split import (
    build_a5c_breadth_split,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import LayerFragmentRows
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


class _TokenLocalFixtureHead:
    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        score = hidden_states[..., 0] + hidden_states[..., -1]
        return torch.stack((score, -score, 0.25 * score.square()), dim=-1)


class _CrossTokenFixtureHead:
    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        shared = hidden_states[..., -1].mean(dim=1, keepdim=True)
        score = hidden_states[..., 0] + shared
        return torch.stack((score, -score, score.square()), dim=-1)


def _directed_locality_receipt(
    probe_name: str, *, cross_token: bool = False
) -> dict[str, object]:
    rows = (
        torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10.0
    ).contiguous()
    adapter = _CrossTokenFixtureHead() if cross_token else _TokenLocalFixtureHead()
    return dict(
        locality_module.audit_same_shape_off_row_locality(
            adapter=adapter,  # type: ignore[arg-type]
            rows=rows,
            probe_name=probe_name,
            absolute_tolerance=report._A5C_TOKEN_LOCALITY_ATOL,
            relative_tolerance=report._A5C_TOKEN_LOCALITY_RTOL,
            solver_authorization=True,
        ).receipt
    )


def _descriptor(seed: int, *, post_delta: bool) -> dict[str, object]:
    return {
        "layer17_graph_sha256": _digest(seed),
        "layer17_lowering_sha256_by_node": {
            f"node-{index}": _digest(seed + index + 1) for index in range(4)
        },
        "composition_graph_sha256": _digest(seed + 5),
        "layer17_parameter_count": 163_094,
        "layer17_macs_per_token": 160_352,
        "composition_parameter_count": 295_129,
        "composition_macs_per_token": 289_600,
        "layer17_post_feedforward_delta_layer_ordinals": (
            [17] if post_delta else []
        ),
        "composition_post_feedforward_delta_layer_ordinals": (
            [17] if post_delta else []
        ),
    }


def _canonical_frozen_descriptor() -> dict[str, object]:
    descriptor = _descriptor(100, post_delta=False)
    descriptor["layer17_graph_sha256"] = report._EXPECTED_LAYER17_GRAPH_SHA256
    descriptor["layer17_lowering_sha256_by_node"] = dict(
        report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
    )
    descriptor["composition_graph_sha256"] = (
        report._EXPECTED_PRIMARY_COMPOSITION_GRAPH_SHA256
    )
    return descriptor


def _metric(
    native_nll: float,
    delta: float,
    kl: float,
    top1: float,
) -> dict[str, float]:
    return {
        "nll_per_token": native_nll + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _condition(
    graph_sha256: str,
    native_nll: float,
    delta: float,
    kl: float,
    top1: float,
) -> dict[str, object]:
    return {
        "graph_sha256": graph_sha256,
        **_metric(native_nll, delta, kl, top1),
    }


def _summary(value: float) -> dict[str, float]:
    return {
        "sum": value,
        "mean": value,
        "minimum": value,
        "median": value,
        "maximum": value,
    }


def _error(value: float) -> dict[str, float]:
    return {
        "rmse": value,
        "reference_rms": 1.0,
        "nrmse": value,
        "max_abs_error": value,
    }


def _target_evidence() -> dict[str, object]:
    teacher_locality = _directed_locality_receipt("native_teacher")
    a4_locality = _directed_locality_receipt("a4_euclidean_baseline")

    def locality() -> dict[str, object]:
        teacher = copy.deepcopy(teacher_locality)
        a4 = copy.deepcopy(a4_locality)
        return {
            "policy": report._A5C_TOKEN_LOCALITY_POLICY,
            "method": report._A5C_TOKEN_LOCALITY_METHOD,
            "probe_states": [
                "native_teacher",
                "a4_euclidean_baseline",
            ],
            "row_count": 8,
            "nontrivial_multirow_probe": True,
            "absolute_tolerance": report._A5C_TOKEN_LOCALITY_ATOL,
            "relative_tolerance": report._A5C_TOKEN_LOCALITY_RTOL,
            "teacher": teacher,
            "a4_baseline": a4,
            "directed_receipt_sha256_by_probe": {
                "teacher": teacher["receipt_sha256"],
                "a4_baseline": a4["receipt_sha256"],
            },
            "native_teacher_baseline_logits_reused_by_solver": True,
            "changes_solver_authorization": True,
            "passed": True,
        }

    chunks = []
    for index in range(14):
        start = index * 8
        chunks.append(
            {
                "chunk_index": index,
                "row_start": start,
                "row_stop": start + 8,
                "row_count": 8,
                "initial_kl": _summary(0.5),
                "selected_kl": _summary(0.4),
                "selected_step": _summary(2.0),
                "trust_projection_count": _summary(0.0),
                "token_locality": locality(),
                "full_solver_receipt_sha256": _digest(300 + index),
            }
        )
    payload = {
        "schema": "fisher_graph.gemma3_l10_l17_a5b_batched_capacity.v1",
        "objective": (
            "independent_per_token_exact_native_to_candidate_kl_through_"
            "adapter_project_logits"
        ),
        "scientific_method": "a5a_frozen_affine_capacity_oracle",
        "throughput_change_only": True,
        "teacher_boundary": "captured_native_layer17_output",
        "candidate_formula": (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        ),
        "initialization": "float64_affine_sum_svd_pseudoinverse_minimum_norm",
        "canonical_target_dtype": "torch.float64",
        "affine_arithmetic_dtype": "torch.float64",
        "coordinate_layout": "joint_concatenated_four_node_rank_182",
        "runtime_correction_dtype": "torch.float32",
        "runtime_correction_cast_count_per_materialization": 1,
        "initial_correction_bit_identical_to_a4_float64_one_cast": True,
        "row_count": 112,
        "row_chunk_size": 8,
        "chunk_count": 14,
        "batching": {
            "one_batched_head_callback_per_optimizer_evaluation": True,
            "independent_adam_parameter_group_per_token": True,
            "independent_kl_best_checkpoint_per_token": True,
            "token_locality_audited_on_native_and_a4_states": True,
            "token_locality_policy": report._A5C_TOKEN_LOCALITY_POLICY,
            "token_locality_absolute_tolerance": (
                report._A5C_TOKEN_LOCALITY_ATOL
            ),
            "token_locality_relative_tolerance": (
                report._A5C_TOKEN_LOCALITY_RTOL
            ),
        },
        "solver": {
            "steps": 64,
            "learning_rate_fraction_of_per_token_initial_coefficient_rms": (
                1.0e-2
            ),
            "minimum_scale_for_zero_rms": 1.0e-12,
            "initial_coefficient_rms": _summary(1.0),
            "effective_learning_rate": _summary(1.0e-2),
            "scale_is_independent_for_each_token": True,
            "ridge": 0.0,
            "trust_radius": None,
            "initial_point_evaluated_as_safe_abstention": True,
        },
        "initial_kl": _summary(0.5),
        "selected_kl": _summary(0.4),
        "absolute_mean_kl_improvement": 0.1,
        "selected_not_worse_than_initial_for_every_token": True,
        "selected_step": _summary(2.0),
        "trust_projection_count": _summary(0.0),
        "initial_state_error": _error(0.5),
        "selected_state_error": _error(0.4),
        "hashes": {
            "initial_coefficient_sha256": _digest(20),
            "selected_coefficient_sha256": _digest(21),
            "initial_correction_sha256": _digest(22),
            "selected_correction_sha256": _digest(23),
            "initial_state_sha256": _digest(24),
            "selected_state_sha256": _digest(25),
        },
        "chunk_receipts": chunks,
        "frozen_affine_membership_by_construction": True,
        "basis_mean_or_decoder_changed": False,
        "deployable_generator_fitted": False,
        "contains_tensor_payloads": False,
    }
    return {
        **payload,
        "receipt_sha256": report._sha256(
            report._TARGET_RECEIPT_DOMAIN, payload
        ),
    }


def _coordinate_row_bank_evidence(
    *,
    row_key_sha256: str = _digest(23),
    compiled_inputs_sha256: str = _digest(24),
) -> dict[str, object]:
    aliases = tuple(f"family-{index}" for index in range(7))
    nodes = tuple(f"node-{index}" for index in range(4))
    ranks = (48, 38, 48, 48)
    fragments = {name: f"fragment-{index}" for index, name in enumerate(nodes)}
    role_fisher = {
        name: {
            "fragment_id": fragments[name],
            "total_mass": 1.0,
            "family_mass_by_alias": {
                alias: 1.0 / len(aliases) for alias in aliases
            },
            "unit_total_mass": True,
            "equal_family_mass": True,
        }
        for name in nodes
    }

    def row_receipt(seed: int, observations: int, sequences: int) -> dict[str, object]:
        return {
            "row_key_sha256": _digest(seed),
            "observations": observations,
            "sequences": sequences,
            "fragment_tensor_sha256s": {
                fragment: {
                    "inputs_sha256": _digest(seed + index * 3 + 1),
                    "contributions_sha256": _digest(seed + index * 3 + 2),
                    "fisher_weights_sha256": _digest(seed + index * 3 + 3),
                }
                for index, fragment in enumerate(fragments.values())
            },
        }

    payload = {
        "schema": (
            report.GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA
        ),
        "format_version": (
            report.GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION
        ),
        "scientific_role": (
            "calibration_a_fit_outer_training_downstream_coordinate_targets"
        ),
        "source_safe": True,
        "contains_tensors": False,
        "contains_prompt_text": False,
        "contains_prompt_identities": False,
        "contains_token_ids": False,
        "heldout_confirmation": False,
        "outer_split": {
            "training_family_aliases": aliases,
            "held_family_alias": "family-held",
            "held_family_rows_accepted": False,
            "held_family_used_for_fit_or_audit": False,
        },
        "authentication": {
            "compiled_inputs_sha256": compiled_inputs_sha256,
            "selected_joint_coordinates_sha256": _digest(21),
            "joint_coordinate_width": 182,
        },
        "frozen_affine_image": {
            "node_order": nodes,
            "rank_by_node": ranks,
            "coordinate_slices": {
                name: {"start": start, "stop": start + rank, "rank": rank}
                for name, start, rank in zip(
                    nodes, (0, 48, 86, 134), ranks, strict=True
                )
            },
            "fragment_id_by_node": fragments,
            "basis_sha256_by_node": {
                name: _digest(40 + index) for index, name in enumerate(nodes)
            },
            "mean_sha256_by_node": {
                name: _digest(50 + index) for index, name in enumerate(nodes)
            },
            "encoder_sha256_by_node": {
                name: _digest(60 + index) for index, name in enumerate(nodes)
            },
            "decoder_sha256_by_node": {
                name: _digest(70 + index) for index, name in enumerate(nodes)
            },
            "basis_artifacts_unchanged_after_decode": True,
            "mean_tensors_byte_identical_after_decode": True,
            "encoder_tensors_byte_identical_after_decode": True,
            "decoder_tensors_byte_identical_after_decode": True,
        },
        "joint_roundtrip_audit": {
            "definition": (
                "sum_node_decode(coordinate_slice)_equals_"
                "sum_node_mean_plus_joint_coordinate_times_"
                "concatenated_frozen_decoder"
            ),
            "joint_roundtrip_sha256": _digest(80),
            "summed_decoded_contribution_sha256": _digest(81),
            "max_abs_difference": 0.0,
            "rms_difference": 0.0,
            "relative_tolerance": 1.0e-11,
            "absolute_tolerance": 1.0e-11,
            "passed": True,
        },
        "inner_split": {
            "method": (
                "domain_separated_hash_rank_per_training_family_"
                "then_preserve_source_row_order"
            ),
            "inner_split_binding_sha256": _digest(82),
            "inner_audit_examples_per_family": 1,
            "fit_example_count": 21,
            "audit_example_count": 7,
            "fit_example_membership_sha256": _digest(83),
            "audit_example_membership_sha256": _digest(84),
            "row_overlap_count": 0,
            "example_overlap_count": 0,
            "rows_exactly_partitioned": True,
            "examples_exactly_partitioned": True,
        },
        "fisher_normalization": {
            "all_rows_preserve_raw_authenticated_fisher_weights": True,
            "fit_and_audit_normalized_independently": True,
            "audit_weights_influence_fit_normalization": False,
            "policy": (
                "equal_total_mass_per_outer_training_family_per_role_and_node"
            ),
            "training_family_count": 7,
            "target_total_mass_per_role_and_node": 1.0,
            "target_mass_per_family": 1.0 / 7.0,
            "fit_by_node": role_fisher,
            "audit_by_node": copy.deepcopy(role_fisher),
        },
        "row_accounting": {
            "all": {
                **row_receipt(23, 112, 28),
                "row_key_sha256": row_key_sha256,
            },
            "fit": row_receipt(90, 84, 21),
            "audit": row_receipt(110, 28, 7),
            "all_observations_equal_fit_plus_audit": True,
            "all_examples_equal_fit_plus_audit": True,
        },
        "consumer_contract": {
            "compatible_with": "fit_frozen_basis_coordinate_generators",
            "contribution_target": (
                "frozen_basis_decode_of_downstream_selected_coordinate_slice"
            ),
            "generator_fit_performed": False,
        },
    }
    return {
        **payload,
        "receipt_sha256": report._sha256(
            report._COORDINATE_ROW_BANK_RECEIPT_DOMAIN, payload
        ),
    }


def _breadth_evidence(bridge_sha256: str) -> dict[str, object]:
    aliases = tuple(f"family-{index}" for index in range(7))
    ownership = {
        f"{alias}/example-{example}": alias
        for alias in aliases
        for example in range(4)
    }
    row_keys = tuple(
        (f"{alias}/example-{example}", position)
        for position in range(4)
        for example in range(4)
        for alias in aliases
    )
    inputs = (
        torch.arange(112 * 5, dtype=torch.float64).reshape(112, 5) / 100.0
    ).contiguous()
    fragments = tuple(f"fragment-{index}" for index in range(4))
    node_order = tuple(f"node-{index}" for index in range(4))
    rows = AlignedFragmentRows(
        rows_by_fragment={
            fragment: LayerFragmentRows(
                inputs=inputs,
                contributions=(inputs * (index + 1.0)).contiguous(),
                fisher_weights=torch.linspace(
                    1.0 + index, 2.0 + index, 112, dtype=torch.float64
                ),
                sequences=28,
            )
            for index, fragment in enumerate(fragments)
        },
        row_keys=row_keys,
    )
    return build_a5c_breadth_split(
        rows=rows,
        compiled_inputs=inputs,
        row_keys=row_keys,
        family_alias_by_example=ownership,
        training_family_aliases=aliases,
        outer_held_family_alias="family-held",
        split_binding_sha256=_digest(400),
        audit_examples_per_family=1,
        node_order=node_order,
        fragment_id_by_node=dict(zip(node_order, fragments, strict=True)),
        source_bridge_receipt_sha256=bridge_sha256,
    ).receipt()


def _cv_evidence(
    *,
    breadth_sha256: str,
    breadth_source: dict[str, object],
    target: dict[str, object],
    fallback: bool,
    frozen: dict[str, object],
    selected: dict[str, object],
) -> dict[str, object]:
    aliases = tuple(f"family-{index}" for index in range(7))
    node_names = tuple(f"node-{index}" for index in range(4))
    lowerings = dict(frozen["layer17_lowering_sha256_by_node"])
    cast_payload = {
        "policy": (
            "cast_compiled_generator_input_to_candidate_state_dtype_before_"
            "factor_casts_and_both_generator_matmuls"
        ),
        "source_dtype": "torch.float64",
        "runtime_dtype": "torch.float32",
        "absolute_tolerance": cv_module._INPUT_CAST_ATOL,
        "relative_tolerance": cv_module._INPUT_CAST_RTOL,
        "source_input_sha256": breadth_source["compiled_inputs_sha256"],
        "runtime_input_sha256": _digest(511),
        "max_abs_cast_error": 0.0,
        "rms_cast_error": 0.0,
        "passed": True,
    }
    input_cast = {
        **cast_payload,
        "lineage_sha256": cv_module._domain_sha256(
            cv_module._INPUT_CAST_LINEAGE_DOMAIN, cast_payload
        ),
    }
    source = {
        "bridge_receipt_sha256": breadth_sha256,
        "source_graph_sha256": frozen["layer17_graph_sha256"],
        "source_model_sha256": report._EXPECTED_MODEL_FINGERPRINT,
        "source_lowering_sha256_by_node": lowerings,
        "native_block_states_sha256": _digest(514),
        "frozen_compiled_block_states_sha256": _digest(515),
        "compiled_correction_base_states_sha256": _digest(516),
        "all_rows_key_sha256": breadth_source["row_key_sha256"],
        "bridge_compiled_inputs_sha256": breadth_source[
            "bridge_compiled_inputs_sha256"
        ],
        "all_rows_input_sha256": breadth_source["compiled_inputs_sha256"],
        "candidate_runtime_input_cast": input_cast,
        "final_head_token_locality_lineage_sha256": (
            report.a5c_target_token_locality_lineage_sha256(target)
        ),
    }
    grid = cv_module.A5C_RIDGE_GRID
    candidates = []
    for ridge_index, ridge in enumerate(grid):
        candidate_kl = 0.6 if fallback else (0.2 if ridge == 1.0e-2 else 0.4)
        candidate_state = candidate_kl + 0.1
        folds = []
        for alias_index, alias in enumerate(aliases):
            fit_key = _digest(520 + alias_index)
            eval_key = _digest(530 + alias_index)
            fit_split = cv_module._domain_sha256(
                cv_module._SPLIT_DOMAIN,
                {
                    "bridge_receipt_sha256": breadth_sha256,
                    "ridge_hex": ridge.hex(),
                    "held_inner_family_alias": alias,
                    "role": "inner_fit_six_families",
                    "row_key_sha256": fit_key,
                    "observations": 96,
                    "sequences": 24,
                },
            )
            eval_split = cv_module._domain_sha256(
                cv_module._SPLIT_DOMAIN,
                {
                    "bridge_receipt_sha256": breadth_sha256,
                    "ridge_hex": ridge.hex(),
                    "held_inner_family_alias": alias,
                    "role": "inner_evaluation_one_family",
                    "row_key_sha256": eval_key,
                    "observations": 16,
                    "sequences": 4,
                },
            )
            folds.append(
                {
                    "held_inner_family_alias": alias,
                    "training_family_count": 6,
                    "fit_observations": 96,
                    "fit_examples": 24,
                    "evaluation_observations": 16,
                    "evaluation_examples": 4,
                    "fit_membership_sha256": _digest(540 + alias_index),
                    "evaluation_membership_sha256": _digest(550 + alias_index),
                    "fit_row_key_sha256": fit_key,
                    "evaluation_row_key_sha256": eval_key,
                    "fit_split_sha256": fit_split,
                    "evaluation_split_sha256": eval_split,
                    "fit_rows_removed_for_signature_overlap": 0,
                    "input_signature_overlap_count": 0,
                    "input_signatures_disjoint": True,
                    "graph_sha256": _digest(560 + ridge_index * 7 + alias_index),
                    "lowering_sha256_by_node": lowerings,
                    "candidate": {
                        "fisher_weighted_final_head_kl": candidate_kl,
                        "fisher_weighted_state_nrmse": candidate_state,
                    },
                    "frozen_baseline": {
                        "fisher_weighted_final_head_kl": 0.5,
                        "fisher_weighted_state_nrmse": 0.6,
                    },
                }
            )
        candidates.append(
            {
                "ridge": ridge,
                "ridge_hex": ridge.hex(),
                "inner_fold_count": 7,
                "family_equal_candidate_final_head_kl": candidate_kl,
                "family_equal_frozen_final_head_kl": 0.5,
                "family_equal_candidate_state_nrmse": candidate_state,
                "family_equal_frozen_state_nrmse": 0.6,
                "folds": folds,
            }
        )
    winner_ridge = 0.0 if fallback else 1.0e-2
    winner_kl = 0.6 if fallback else 0.2
    selected_ridge = None if fallback else winner_ridge
    selection = {
        "objective": cv_module._SELECTION_OBJECTIVE,
        "winner_ridge_before_fallback": winner_ridge,
        "winner_ridge_hex_before_fallback": winner_ridge.hex(),
        "winner_family_equal_final_head_kl": winner_kl,
        "frozen_family_equal_final_head_kl": 0.5,
        "absolute_kl_improvement": 0.5 - winner_kl,
        "best_ridge_strictly_improves_frozen": not fallback,
        "use_frozen_fallback": fallback,
        "selected_ridge": selected_ridge,
        "selected_ridge_hex": None if fallback else winner_ridge.hex(),
    }
    final_refit = (
        {
            "performed": False,
            "selected_ridge": None,
            "fit_uses_all_outer_training_examples": False,
            "fit_observations": 0,
            "fit_examples": 0,
            "fit_fisher_normalization": "not_applicable_frozen_fallback",
            "descriptive_eval_source": "not_applicable_frozen_fallback",
            "descriptive_eval_is_subset_of_final_fit": False,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": None,
            "descriptive_eval_split_sha256": None,
            "graph_sha256": None,
            "lowering_sha256_by_node": {},
        }
        if fallback
        else {
            "performed": True,
            "selected_ridge": winner_ridge,
            "fit_uses_all_outer_training_examples": True,
            "fit_observations": 112,
            "fit_examples": 28,
            "fit_fisher_normalization": (
                "equal_total_mass_per_outer_training_family_and_node"
            ),
            "descriptive_eval_source": (
                "bridge_audit_rows_rekeyed_after_becoming_subset_of_all_rows_fit"
            ),
            "descriptive_eval_is_subset_of_final_fit": True,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": _digest(700),
            "descriptive_eval_split_sha256": _digest(701),
            "graph_sha256": selected["layer17_graph_sha256"],
            "lowering_sha256_by_node": dict(
                selected["layer17_lowering_sha256_by_node"]
            ),
        }
    )
    payload = {
        "schema": cv_module.GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_SCHEMA,
        "format_version": 1,
        "scientific_role": (
            "outer_training_only_nested_family_disjoint_ridge_selection"
        ),
        "source": source,
        "configuration": {
            "generator_rank": 16,
            "ridge_grid": list(grid),
            "ridge_grid_hex": [ridge.hex() for ridge in grid],
            "inner_fold_count": 7,
            "inner_training_family_count": 6,
            "inner_evaluation_family_count": 1,
            "selection_objective": cv_module._SELECTION_OBJECTIVE,
            "state_diagnostic": cv_module._STATE_DIAGNOSTIC,
            "fallback_rule": cv_module._FALLBACK_RULE,
            "output_boundary": "layer.17.mlp.delta",
            "duplicate_cross_split_input_policy": (
                "exclude_from_inner_fit_every_row_whose_compiled_input_signature_"
                "occurs_in_that_folds_evaluation_family_then_require_zero_overlap"
            ),
            "candidate_generator_execution_dtype": "torch.float32",
            "candidate_generator_execution_policy": (
                "cast_inputs_then_cast_generator_factors_to_input_dtype_before_"
                "both_matmuls_matching_modal_graph_executor"
            ),
            "final_head_chunk_rows": 8,
            "final_head_chunking": (
                "fixed_row_chunks_over_token_local_final_norm_and_lm_head_"
                "with_float64_full_vocabulary_kl_accumulation"
            ),
        },
        "ownership": {
            "outer_training_family_aliases": aliases,
            "outer_held_family_alias": "family-held",
            "outer_training_family_count": 7,
            "outer_training_example_count": 28,
            "outer_training_observation_count": 112,
            "outer_training_membership_sha256": _digest(710),
            "outer_held_family_present_in_ownership": False,
            "outer_held_family_states_or_rows_accessed": False,
        },
        "candidates": candidates,
        "selection": selection,
        "final_refit": final_refit,
        "safety": dict(cv_module._SAFETY),
    }
    return {
        **payload,
        "receipt_sha256": cv_module._domain_sha256(
            cv_module._REPORT_DOMAIN, payload
        ),
    }


def _evidence_receipts(
    *,
    fallback: bool,
    frozen: dict[str, object],
    selected: dict[str, object],
) -> dict[str, object]:
    target = _target_evidence()
    bridge = _coordinate_row_bank_evidence()
    breadth = _breadth_evidence(str(bridge["receipt_sha256"]))
    breadth_source = breadth["source"]
    assert isinstance(breadth_source, dict)
    bridge = _coordinate_row_bank_evidence(
        row_key_sha256=str(breadth_source["row_key_sha256"]),
        compiled_inputs_sha256=str(
            breadth_source["bridge_compiled_inputs_sha256"]
        ),
    )
    breadth = _breadth_evidence(str(bridge["receipt_sha256"]))
    breadth_source = breadth["source"]
    assert isinstance(breadth_source, dict)
    cv = _cv_evidence(
        breadth_sha256=str(breadth["receipt_sha256"]),
        breadth_source=breadth_source,
        target=target,
        fallback=fallback,
        frozen=frozen,
        selected=selected,
    )
    return {
        "target_solve": target,
        "coordinate_row_bank": bridge,
        "breadth_split": breadth,
        "ridge_cv": cv,
    }


def _inputs(*, fallback: bool = False) -> dict[str, object]:
    native_nll = 2.0
    frozen = _canonical_frozen_descriptor()
    if fallback:
        selected = copy.deepcopy(frozen)
    else:
        selected = _descriptor(200, post_delta=True)
        selected["layer17_lowering_sha256_by_node"] = {
            name: _digest(201 + index)
            for index, name in enumerate(
                report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
            )
        }
    ridge = None if fallback else 1.0e-2
    evidence = _evidence_receipts(
        fallback=fallback,
        frozen=frozen,
        selected=selected,
    )
    target = report._compact_target_evidence(evidence["target_solve"])
    row_bank = report._compact_coordinate_row_bank_evidence(
        evidence["coordinate_row_bank"]
    )
    breadth = report._compact_breadth_evidence(evidence["breadth_split"])
    ridge_cv = report._compact_cv_evidence(evidence["ridge_cv"])
    lineage = {
        "target_solve_receipt_sha256": target["receipt_sha256"],
        "coordinate_row_bank_receipt_sha256": row_bank["receipt_sha256"],
        "breadth_split_receipt_sha256": breadth["receipt_sha256"],
        "ridge_cv_receipt_sha256": ridge_cv["receipt_sha256"],
        "layer10_graph_sha256": report._EXPECTED_LAYER10_GRAPH_SHA256,
        "layer10_lowering_sha256_by_node": dict(
            report._EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE
        ),
        "matched_double_deletion_graph_sha256": _digest(51),
    }
    kind = "frozen_source_fallback" if fallback else "learned_correction"
    freeze = report.a5c_selection_freeze_sha256(
        kind=kind,
        selected_ridge=ridge,
        lineage=lineage,
        selected=selected,
        frozen_reference=frozen,
    )
    selected_executable = {
        "kind": kind,
        "selected_ridge": ridge,
        "lineage": lineage,
        "selected": selected,
        "frozen_reference": frozen,
        "selection_freeze_sha256": freeze,
    }
    selected_composition = (
        _metric(native_nll, 0.20, 0.30, 0.70)
        if fallback
        else _metric(native_nll, 0.10, 0.20, 0.80)
    )
    selected_layer17 = (
        _metric(native_nll, 0.18, 0.25, 0.72)
        if fallback
        else _metric(native_nll, 0.07, 0.16, 0.84)
    )
    source_scorer = {
        "execution_path": "full_model_logits_fixed_capacity_a3_trajectory_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": 35,
        "logical_valid_tokens": 36,
        "native": {"nll_per_token": native_nll},
        "conditions": {
            "layer10_only": _metric(native_nll, 0.08, 0.09, 0.82),
            "trajectory_corrected_layer17_only": selected_layer17,
            "frozen_uncorrected_composition": _metric(
                native_nll, 0.20, 0.30, 0.70
            ),
            "trajectory_corrected_composition": selected_composition,
            "matched_double_deletion": _metric(
                native_nll, 0.90, 1.10, 0.40
            ),
        },
        "resource_accounting": {
            name: {
                **values,
                "executed_peak_live_modal_width": (
                    0 if name == "matched_double_deletion" else 48
                ),
            }
            for name, values in scorer_module._EXPECTED_CONDITION_RESOURCES.items()
        },
        "exact_resources_match_protocol": True,
        "latency_or_kernel_speed_claim": False,
    }
    graph_hashes = {
        "layer10_only": lineage["layer10_graph_sha256"],
        "selected_layer17_only": selected["layer17_graph_sha256"],
        "frozen_uncorrected_composition": frozen["composition_graph_sha256"],
        "selected_composition": selected["composition_graph_sha256"],
        "matched_double_deletion": lineage[
            "matched_double_deletion_graph_sha256"
        ],
    }
    evaluation = {
        "assessment_role": "calibration_a_outer_family_bounded_development",
        "outer_fold_index": 0,
        "logical_valid_tokens": 36,
        "supervised_tokens": 35,
        "native": {"nll_per_token": native_nll},
        "conditions": {
            "layer10_only": {
                "graph_sha256": graph_hashes["layer10_only"],
                **source_scorer["conditions"]["layer10_only"],
            },
            "selected_layer17_only": {
                "graph_sha256": graph_hashes["selected_layer17_only"],
                **selected_layer17,
            },
            "frozen_uncorrected_composition": _condition(
                str(graph_hashes["frozen_uncorrected_composition"]),
                native_nll,
                0.20,
                0.30,
                0.70,
            ),
            "selected_composition": {
                "graph_sha256": graph_hashes["selected_composition"],
                **selected_composition,
            },
            "matched_double_deletion": _condition(
                str(graph_hashes["matched_double_deletion"]),
                native_nll,
                0.90,
                1.10,
                0.40,
            ),
        },
        "source_scorer_evaluation": source_scorer,
        "source_scorer_receipt_sha256": (
            report.a5c_source_scorer_receipt_sha256(
                source_scorer_evaluation=source_scorer,
                outer_fold_index=0,
                condition_graph_sha256_by_name=graph_hashes,
            )
        ),
        "source_scorer_evaluation_sha256": (
            report.a5c_source_scorer_evaluation_sha256(source_scorer)
        ),
        "resource_accounting_sha256": report.a5c_resource_accounting_sha256(
            source_scorer["resource_accounting"]
        ),
        "resource_accounting_reference": (
            "score_trajectory_correction_fold.resource_accounting"
        ),
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
    }
    evaluation_sha = report.a5c_outer_evaluation_sha256(evaluation)
    return {
        "source_bindings": dict(report._EXPECTED_SOURCE_BINDINGS),
        "runtime": {
            "model_id": report._EXPECTED_MODEL_ID,
            "requested_revision": report._EXPECTED_MODEL_REVISION,
            "model_fingerprint": report._EXPECTED_MODEL_FINGERPRINT,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        "configuration": {
            "outer_fold_index": 0,
            "training_family_count": 7,
            "training_examples_per_family": 4,
            "row_selection_policy": "all_valid_captured_rows_per_example",
            "inner_audit_examples_per_family": 1,
            "target_solver_steps": 64,
            "target_batch_rows": 8,
            "target_learning_rate_fraction": 1.0e-2,
            "target_ridge": 0.0,
            "target_trust_radius": None,
            "generator_rank": 16,
            "ridge_grid": list(cv_module.A5C_RIDGE_GRID),
            "held_examples_scored": 1,
        },
        "capture": {
            "capture_sha256": _digest(10),
            "capture_audit_sha256": _digest(11),
            "source_row_catalog_sha256": row_bank["row_key_sha256"],
            "training_family_count": 7,
            "training_example_count": 28,
            "captured_observation_count": 112,
            "selected_target_row_count": 112,
            "outer_held_family_rows_present": False,
            "all_required_capture_audits_pass": True,
        },
        "target_solve": target,
        "coordinate_row_bank": row_bank,
        "breadth_split": breadth,
        "ridge_cv": ridge_cv,
        "evidence_receipts": evidence,
        "selected_executable": selected_executable,
        "chronology": {
            "ridge_cv_completed_event": 1,
            "executable_frozen_event": 2,
            "outer_held_batch_selected_event": 3,
            "outer_held_model_evaluated_event": 4,
            "outer_held_batch_selected_or_scored_before_freeze": False,
            "executable_frozen_before_outer_held_batch_selection": True,
            "executable_frozen_before_outer_held_model_evaluation": True,
            "ridge_cv_receipt_sha256": ridge_cv["receipt_sha256"],
            "selection_freeze_sha256": freeze,
            "outer_evaluation_sha256": evaluation_sha,
        },
        "outer_evaluation": evaluation,
        "comparison_to_a5b": {
            "a5b_report_sha256": report._EXPECTED_A5B_REPORT_SHA256,
            "same_outer_fold": True,
            "same_held_example_policy": True,
            "a5b_learned_composition": dict(
                report._EXPECTED_A5B_LEARNED_COMPOSITION
            ),
        },
    }


def _build(*, fallback: bool = False) -> dict[str, object]:
    return report.build_gemma3_l10_l17_a5c_report(**_inputs(fallback=fallback))


def _rehash(value: dict[str, object]) -> None:
    value.pop("report_sha256", None)
    value["report_sha256"] = report._sha256(report._REPORT_DOMAIN, value)


def _refresh_freeze(value: dict[str, object]) -> None:
    executable = value["selected_executable"]
    freeze = report.a5c_selection_freeze_sha256(
        kind=executable["kind"],
        selected_ridge=executable["selected_ridge"],
        lineage=executable["lineage"],
        selected=executable["selected"],
        frozen_reference=executable["frozen_reference"],
    )
    executable["selection_freeze_sha256"] = freeze
    value["chronology"]["selection_freeze_sha256"] = freeze


def _rehash_directed_receipt(value: dict[str, object]) -> None:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = locality_module._receipt_sha256(value)


def _refresh_target_receipt_chain(value: dict[str, object]) -> None:
    target = value["evidence_receipts"]["target_solve"]
    target.pop("receipt_sha256", None)
    target_sha256 = report._sha256(report._TARGET_RECEIPT_DOMAIN, target)
    target["receipt_sha256"] = target_sha256
    value["target_solve"]["receipt_sha256"] = target_sha256
    value["selected_executable"]["lineage"][
        "target_solve_receipt_sha256"
    ] = target_sha256
    _refresh_freeze(value)
    _rehash(value)


def _target_locality(
    value: dict[str, object], *, chunk_index: int = 0
) -> dict[str, object]:
    return value["evidence_receipts"]["target_solve"]["chunk_receipts"][
        chunk_index
    ]["token_locality"]


def _refresh_nested_locality(
    value: dict[str, object], *, role: str, chunk_index: int = 0
) -> None:
    locality = _target_locality(value, chunk_index=chunk_index)
    nested = locality[role]
    _rehash_directed_receipt(nested)
    locality["directed_receipt_sha256_by_probe"][role] = nested[
        "receipt_sha256"
    ]
    _refresh_target_receipt_chain(value)


def test_learned_report_builds_with_derived_conclusion() -> None:
    value = _build()
    assert value["schema"] == report.GEMMA3_L10_L17_A5C_REPORT_SCHEMA
    conclusion = value["conclusion"]
    assert conclusion["use_frozen_fallback"] is False
    assert conclusion["selected_improves_frozen_kl"] is True
    assert conclusion["selected_improves_a5b_learned_kl"] is True
    assert conclusion["does_not_establish_eight_fold_competitive_compilation"] is True


def test_quality_metric_matches_source_scorer_numerical_kl_floor() -> None:
    boundary = _metric(2.0, 0.1, -1.0e-12, 0.75)
    assert report._validate_quality_metric(boundary, label="boundary")[
        "native_to_candidate_kl_per_token"
    ] == -1.0e-12

    below = _metric(2.0, 0.1, -1.0001e-12, 0.75)
    with pytest.raises(ValueError, match="outside its finite range"):
        report._validate_quality_metric(below, label="below")


def test_coordinate_roundtrip_scalar_check_allows_one_ulp_summary_rounding() -> None:
    bridge = _coordinate_row_bank_evidence()
    roundtrip = bridge["joint_roundtrip_audit"]
    roundtrip["max_abs_difference"] = 1.0
    roundtrip["rms_difference"] = math.nextafter(1.0, math.inf)
    payload = dict(bridge)
    payload.pop("receipt_sha256")
    bridge["receipt_sha256"] = report._sha256(
        report._COORDINATE_ROW_BANK_RECEIPT_DOMAIN, payload
    )

    validated = report._validate_coordinate_row_bank_evidence(
        bridge,
        configuration={
            "training_family_count": 7,
            "inner_audit_examples_per_family": 1,
        },
        expected_rows=112,
        expected_examples=28,
    )

    assert validated["joint_roundtrip_audit"]["rms_difference"] > 1.0

    roundtrip["rms_difference"] = 1.0 + 4.0 * math.ulp(1.0)
    payload = dict(bridge)
    payload.pop("receipt_sha256")
    bridge["receipt_sha256"] = report._sha256(
        report._COORDINATE_ROW_BANK_RECEIPT_DOMAIN, payload
    )
    with pytest.raises(ValueError, match="roundtrip pass is contradictory"):
        report._validate_coordinate_row_bank_evidence(
            bridge,
            configuration={
                "training_family_count": 7,
                "inner_audit_examples_per_family": 1,
            },
            expected_rows=112,
            expected_examples=28,
        )


def test_frozen_fallback_requires_exact_selected_descriptor_and_metrics() -> None:
    value = _build(fallback=True)
    assert value["conclusion"]["selected_exactly_matches_frozen"] is True

    metric_tamper = copy.deepcopy(value)
    selected = metric_tamper["outer_evaluation"]["conditions"][
        "selected_composition"
    ]
    selected["native_to_candidate_kl_per_token"] = 0.29
    metric_tamper["chronology"][
        "outer_evaluation_sha256"
    ] = report.a5c_outer_evaluation_sha256(
        metric_tamper["outer_evaluation"]
    )
    metric_tamper["conclusion"] = report.derive_gemma3_l10_l17_a5c_conclusion(
        selected_executable=metric_tamper["selected_executable"],
        outer_evaluation=metric_tamper["outer_evaluation"],
        comparison_to_a5b=metric_tamper["comparison_to_a5b"],
    )
    _rehash(metric_tamper)
    with pytest.raises(ValueError, match="compact scorer projection drifted"):
        report.validate_gemma3_l10_l17_a5c_report(metric_tamper)

    descriptor_tamper = copy.deepcopy(value)
    descriptor_tamper["selected_executable"]["selected"][
        "composition_graph_sha256"
    ] = _digest(999)
    _rehash(descriptor_tamper)
    with pytest.raises(ValueError, match="fallback executable hashes/resources"):
        report.validate_gemma3_l10_l17_a5c_report(descriptor_tamper)


def test_self_rehashed_chronology_and_conclusion_tampering_is_rejected() -> None:
    chronology = _build()
    chronology["chronology"]["executable_frozen_event"] = 4
    _rehash(chronology)
    with pytest.raises(ValueError, match="freeze-before-held chronology"):
        report.validate_gemma3_l10_l17_a5c_report(chronology)

    conclusion = _build()
    conclusion["conclusion"]["selected_improves_frozen_kl"] = False
    _rehash(conclusion)
    with pytest.raises(ValueError, match="conclusion contradicts"):
        report.validate_gemma3_l10_l17_a5c_report(conclusion)


def test_receipt_lineage_and_freeze_tampering_is_rejected() -> None:
    lineage = _build()
    lineage["selected_executable"]["lineage"][
        "breadth_split_receipt_sha256"
    ] = _digest(998)
    _rehash(lineage)
    with pytest.raises(ValueError, match="lineage contradicts"):
        report.validate_gemma3_l10_l17_a5c_report(lineage)

    freeze = _build()
    freeze["selected_executable"]["selection_freeze_sha256"] = _digest(997)
    freeze["chronology"]["selection_freeze_sha256"] = _digest(997)
    _rehash(freeze)
    with pytest.raises(ValueError, match="freeze hash is contradictory"):
        report.validate_gemma3_l10_l17_a5c_report(freeze)


def test_canonical_executable_is_pinned_beyond_self_consistent_lineage() -> None:
    value = _build()
    value["selected_executable"]["lineage"]["layer10_graph_sha256"] = (
        _digest(996)
    )
    _refresh_freeze(value)
    _rehash(value)
    with pytest.raises(ValueError, match="canonical runtime catalog"):
        report.validate_gemma3_l10_l17_a5c_report(value)


def test_fixed_run_configuration_and_resources_reject_self_rehashed_drift() -> None:
    fold = _build()
    fold["configuration"]["outer_fold_index"] = 1
    _rehash(fold)
    with pytest.raises(ValueError, match="fixed configuration drifted"):
        report.validate_gemma3_l10_l17_a5c_report(fold)

    solver = _build()
    solver["configuration"]["target_solver_steps"] = 65
    _rehash(solver)
    with pytest.raises(ValueError, match="fixed configuration drifted"):
        report.validate_gemma3_l10_l17_a5c_report(solver)

    grid = _build()
    grid["configuration"]["ridge_grid"] = [0.0, 1.0e-4, 1.0]
    _rehash(grid)
    with pytest.raises(ValueError, match="fixed canonical grid"):
        report.validate_gemma3_l10_l17_a5c_report(grid)

    resources = _build()
    resources["selected_executable"]["selected"][
        "composition_parameter_count"
    ] += 1
    _rehash(resources)
    with pytest.raises(ValueError, match="canonical executor"):
        report.validate_gemma3_l10_l17_a5c_report(resources)


def test_shared_bridge_input_identity_uses_the_correct_hash_domain() -> None:
    value = _build()
    evidence = value["evidence_receipts"]
    bridge_sha = evidence["coordinate_row_bank"]["authentication"][
        "compiled_inputs_sha256"
    ]
    breadth_source = evidence["breadth_split"]["source"]
    cv_source = evidence["ridge_cv"]["source"]

    assert breadth_source["compiled_inputs_sha256"] != bridge_sha
    assert cv_source["all_rows_input_sha256"] != bridge_sha
    assert breadth_source["bridge_compiled_inputs_sha256"] == bridge_sha
    assert cv_source["bridge_compiled_inputs_sha256"] == bridge_sha

    capture = copy.deepcopy(value)
    capture["capture"]["source_row_catalog_sha256"] = _digest(995)
    _rehash(capture)
    with pytest.raises(ValueError, match="evidence receipt lineage"):
        report.validate_gemma3_l10_l17_a5c_report(capture)


def test_directed_locality_rejects_missing_pair_after_full_rehash() -> None:
    value = _build()
    locality = _target_locality(value)
    locality["teacher"]["directed_pair_checks"].pop()
    _refresh_nested_locality(value, role="teacher")
    with pytest.raises(ValueError, match="receipt structure drifted"):
        report.validate_gemma3_l10_l17_a5c_report(value)


def test_directed_locality_rejects_changed_role_or_method_after_rehash() -> None:
    role = _build()
    locality = _target_locality(role)
    locality["teacher"]["scientific_role"] = (
        "diagnostic_only_not_solver_authorization"
    )
    locality["teacher"]["changes_solver_authorization"] = False
    _refresh_nested_locality(role, role="teacher")
    with pytest.raises(ValueError, match="exact passing A5c directed audit"):
        report.validate_gemma3_l10_l17_a5c_report(role)

    method = _build()
    _target_locality(method)["method"] = "legacy_singleton_method"
    _refresh_target_receipt_chain(method)
    with pytest.raises(ValueError, match="directed policy or row contract"):
        report.validate_gemma3_l10_l17_a5c_report(method)


def test_directed_locality_rejects_failing_pair_or_wrong_chunk_rows() -> None:
    failing = _build()
    locality = _target_locality(failing)
    failed_receipt = _directed_locality_receipt(
        "native_teacher", cross_token=True
    )
    assert failed_receipt["passed"] is False
    locality["teacher"] = failed_receipt
    locality["directed_receipt_sha256_by_probe"]["teacher"] = failed_receipt[
        "receipt_sha256"
    ]
    _refresh_target_receipt_chain(failing)
    with pytest.raises(ValueError, match="exact passing A5c directed audit"):
        report.validate_gemma3_l10_l17_a5c_report(failing)

    rows = _build()
    _target_locality(rows)["row_count"] = 7
    _refresh_target_receipt_chain(rows)
    with pytest.raises(ValueError, match="directed policy or row contract"):
        report.validate_gemma3_l10_l17_a5c_report(rows)


@pytest.mark.parametrize(
    "tamper",
    ("extra_field", "bool_ratio", "equal_source_hash", "weak_mutation"),
)
def test_directed_locality_rejects_loose_or_forged_nested_evidence(
    tamper: str,
) -> None:
    value = _build()
    locality = _target_locality(value)
    teacher = locality["teacher"]
    if tamper == "extra_field":
        teacher["unexpected"] = "loose"
    elif tamper == "bool_ratio":
        teacher["directed_pair_checks"][0]["max_abs_over_allowed"] = True
    elif tamper == "equal_source_hash":
        source = teacher["source_counterfactuals"][0]
        source["mutated_source_row_sha256"] = source["source_row_sha256"]
    else:
        teacher["source_counterfactuals"][0][
            "minimum_absolute_coordinate_change"
        ] = 0.0
    _refresh_nested_locality(value, role="teacher")
    with pytest.raises(ValueError):
        report.validate_gemma3_l10_l17_a5c_report(value)


def test_tensor_and_source_sensitive_payloads_are_rejected() -> None:
    tensor = _build()
    tensor["capture"]["raw_rows"] = torch.zeros(1)
    with pytest.raises(TypeError, match="tensor payload"):
        report.validate_gemma3_l10_l17_a5c_report(tensor)

    prompt = _build()
    prompt["capture"]["prompt_text"] = "secret source text"
    _rehash(prompt)
    with pytest.raises(ValueError, match="prohibited source data"):
        report.validate_gemma3_l10_l17_a5c_report(prompt)


def test_canonical_a5b_binding_is_required_even_after_rehash() -> None:
    value = _build()
    value["source_bindings"]["a5b_report_sha256"] = _digest(996)
    _rehash(value)
    with pytest.raises(ValueError, match="canonical A5b"):
        report.validate_gemma3_l10_l17_a5c_report(value)

    inherited = _build()
    inherited["source_bindings"]["protocol_sha256"] = _digest(995)
    _rehash(inherited)
    with pytest.raises(ValueError, match="canonical A5b"):
        report.validate_gemma3_l10_l17_a5c_report(inherited)


def test_canonical_a5b_comparison_is_required_even_after_rehash() -> None:
    value = _build()
    value["comparison_to_a5b"]["a5b_learned_composition"][
        "native_to_candidate_kl_per_token"
    ] = 0.0
    value["conclusion"] = report.derive_gemma3_l10_l17_a5c_conclusion(
        selected_executable=value["selected_executable"],
        outer_evaluation=value["outer_evaluation"],
        comparison_to_a5b=value["comparison_to_a5b"],
    )
    _rehash(value)
    with pytest.raises(ValueError, match="metrics differ from canonical A5b"):
        report.validate_gemma3_l10_l17_a5c_report(value)


def test_source_scorer_receipt_is_nonempty_and_raw_resources_are_validated() -> None:
    value = _build()
    receipt = value["outer_evaluation"]["source_scorer_receipt_sha256"]
    assert isinstance(receipt, str) and len(receipt) == 64

    tampered = copy.deepcopy(value)
    resources = tampered["outer_evaluation"]["source_scorer_evaluation"][
        "resource_accounting"
    ]
    resources["layer10_only"]["graph_parameters"] += 1
    tampered["chronology"][
        "outer_evaluation_sha256"
    ] = report.a5c_outer_evaluation_sha256(tampered["outer_evaluation"])
    _rehash(tampered)
    with pytest.raises(ValueError, match="resources drifted"):
        report.validate_gemma3_l10_l17_a5c_report(tampered)


def test_evidence_semantics_and_cv_freeze_crosslinks_survive_self_rehash() -> None:
    target = _build()
    target_evidence = target["evidence_receipts"]["target_solve"]
    target_evidence["teacher_boundary"] = "invented_boundary"
    target_payload = dict(target_evidence)
    target_payload.pop("receipt_sha256")
    target_receipt = report._sha256(report._TARGET_RECEIPT_DOMAIN, target_payload)
    target_evidence["receipt_sha256"] = target_receipt
    target["target_solve"]["receipt_sha256"] = target_receipt
    target["selected_executable"]["lineage"][
        "target_solve_receipt_sha256"
    ] = target_receipt
    _refresh_freeze(target)
    _rehash(target)
    with pytest.raises(ValueError, match="target evidence contract drifted"):
        report.validate_gemma3_l10_l17_a5c_report(target)

    cv = _build()
    cv_evidence = cv["evidence_receipts"]["ridge_cv"]
    cv_evidence["source"]["source_graph_sha256"] = _digest(994)
    cv_payload = dict(cv_evidence)
    cv_payload.pop("receipt_sha256")
    cv_receipt = cv_module._domain_sha256(cv_module._REPORT_DOMAIN, cv_payload)
    cv_evidence["receipt_sha256"] = cv_receipt
    cv["ridge_cv"]["receipt_sha256"] = cv_receipt
    cv["selected_executable"]["lineage"][
        "ridge_cv_receipt_sha256"
    ] = cv_receipt
    cv["chronology"]["ridge_cv_receipt_sha256"] = cv_receipt
    _refresh_freeze(cv)
    _rehash(cv)
    with pytest.raises(ValueError, match="evidence receipt lineage"):
        report.validate_gemma3_l10_l17_a5c_report(cv)


def test_executor_application_semantics_are_part_of_the_freeze() -> None:
    value = _build()
    value["selected_executable"]["selected"][
        "composition_post_feedforward_delta_layer_ordinals"
    ] = []
    _rehash(value)
    with pytest.raises(ValueError, match="application semantics drifted"):
        report.validate_gemma3_l10_l17_a5c_report(value)


def test_all_valid_row_and_fully_removed_example_accounting_is_exact() -> None:
    capture = _build()
    capture["capture"]["captured_observation_count"] = 113
    _rehash(capture)
    with pytest.raises(ValueError, match="capture accounting"):
        report.validate_gemma3_l10_l17_a5c_report(capture)

    breadth = _build()
    breadth["breadth_split"]["fit_example_count"] = 20
    _rehash(breadth)
    with pytest.raises(ValueError, match="breadth split"):
        report.validate_gemma3_l10_l17_a5c_report(breadth)


def test_save_load_roundtrip_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    value = _build()
    destination = tmp_path / "a5c.json"
    saved = report.save_gemma3_l10_l17_a5c_report(destination, value)
    assert report.load_gemma3_l10_l17_a5c_report(destination) == saved
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report.save_gemma3_l10_l17_a5c_report(destination, value)
    assert not tuple(tmp_path.glob(".a5c.json.*.tmp"))


def test_load_rejects_nonfinite_json(tmp_path: Path) -> None:
    destination = tmp_path / "nan.json"
    destination.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        report.load_gemma3_l10_l17_a5c_report(destination)


def test_prepublication_bundle_roundtrips_privately_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a5c.json"
    bundle_path = prepublication.default_a5c_prepublication_bundle_path(
        destination
    )
    inputs = _inputs()

    saved = prepublication.save_a5c_prepublication_bundle(
        bundle_path,
        final_output=destination,
        report_inputs=inputs,
    )

    assert stat.S_IMODE(bundle_path.stat().st_mode) == 0o600
    loaded = prepublication.load_a5c_prepublication_bundle(
        bundle_path, final_output=destination
    )
    assert loaded == saved
    assert loaded["target_report_schema"] == report.GEMMA3_L10_L17_A5C_REPORT_SCHEMA
    assert loaded["target_report_format_version"] == 1
    assert loaded["build_role"] == (
        "exact_tensor_free_inputs_for_build_gemma3_l10_l17_a5c_report"
    )
    assert loaded["report_inputs"] == json.loads(json.dumps(inputs))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepublication.save_a5c_prepublication_bundle(
            bundle_path,
            final_output=destination,
            report_inputs=inputs,
        )
    assert not tuple(tmp_path.glob(".a5c.prepublication-bundle.json.*.tmp"))


def test_prepublication_bundle_rejects_tensors_sensitive_fields_and_tamper(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a5c.json"
    bundle_path = prepublication.default_a5c_prepublication_bundle_path(
        destination
    )
    tensor_inputs = _inputs()
    tensor_inputs["capture"]["payload"] = torch.ones(1)  # type: ignore[index]
    with pytest.raises(TypeError, match="contains a tensor"):
        prepublication.save_a5c_prepublication_bundle(
            bundle_path,
            final_output=destination,
            report_inputs=tensor_inputs,
        )
    sensitive_inputs = _inputs()
    sensitive_inputs["capture"]["token_ids"] = [1, 2]  # type: ignore[index]
    with pytest.raises(ValueError, match="forbidden sensitive field"):
        prepublication.save_a5c_prepublication_bundle(
            bundle_path,
            final_output=destination,
            report_inputs=sensitive_inputs,
        )
    for forbidden_key, forbidden_value in (
        ("labels", [1, 0]),
        ("target_ids", [2, 3]),
        ("source_model_weights", {"layer": [0.5]}),
        ("raw_rows", [[1.0, 2.0]]),
        ("selected_coordinates", [[0.25]]),
    ):
        forbidden_inputs = _inputs()
        forbidden_inputs["capture"][forbidden_key] = (  # type: ignore[index]
            forbidden_value
        )
        with pytest.raises(ValueError, match="forbidden sensitive field"):
            prepublication.save_a5c_prepublication_bundle(
                bundle_path,
                final_output=destination,
                report_inputs=forbidden_inputs,
            )

    prepublication.save_a5c_prepublication_bundle(
        bundle_path,
        final_output=destination,
        report_inputs=_inputs(),
    )
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    raw["report_inputs"]["capture"]["captured_observation_count"] += 1
    bundle_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="report-input hash mismatch"):
        prepublication.load_a5c_prepublication_bundle(
            bundle_path, final_output=destination
        )


def test_prepublication_replay_cli_publishes_then_removes_bundle(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a5c.json"
    bundle_path = prepublication.default_a5c_prepublication_bundle_path(
        destination
    )
    prepublication.save_a5c_prepublication_bundle(
        bundle_path,
        final_output=destination,
        report_inputs=_inputs(),
    )

    assert prepublication.main(
        ["--bundle", str(bundle_path), "--output", str(destination)]
    ) == 0

    assert not bundle_path.exists()
    assert report.load_gemma3_l10_l17_a5c_report(destination)[
        "report_sha256"
    ]


def test_prepublication_replay_preserves_bundle_on_builder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "a5c.json"
    bundle_path = prepublication.default_a5c_prepublication_bundle_path(
        destination
    )
    prepublication.save_a5c_prepublication_bundle(
        bundle_path,
        final_output=destination,
        report_inputs=_inputs(),
    )
    monkeypatch.setattr(
        prepublication,
        "build_gemma3_l10_l17_a5c_report",
        lambda **_: (_ for _ in ()).throw(RuntimeError("builder failed")),
    )

    with pytest.raises(RuntimeError, match="builder failed"):
        prepublication.finalize_a5c_prepublication_bundle(
            bundle_path, output=destination
        )

    assert bundle_path.exists()
    assert not destination.exists()


def test_prepublication_replay_preserves_bundle_when_final_exists(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a5c.json"
    bundle_path = prepublication.default_a5c_prepublication_bundle_path(
        destination
    )
    prepublication.save_a5c_prepublication_bundle(
        bundle_path,
        final_output=destination,
        report_inputs=_inputs(),
    )
    destination.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepublication.finalize_a5c_prepublication_bundle(
            bundle_path, output=destination
        )

    assert bundle_path.exists()
