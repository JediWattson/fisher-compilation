from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a5b_generator_microcanary as micro
import fisher_graph.gemma3_l10_l17_trajectory_correction_lofo as lofo
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    V8_FAMILY_LOFO_FAMILY_ALIASES,
)


def _metric(
    *,
    native_nll: float,
    delta_nll: float,
    kl: float,
    top1: float,
) -> dict[str, float]:
    return {
        "nll_per_token": native_nll + delta_nll,
        "delta_nll_per_token": delta_nll,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _digest(value: int) -> str:
    return f"{value:064x}"


def _summary(count: int, value: float) -> dict[str, float]:
    return {
        "sum": count * value,
        "mean": value,
        "minimum": value,
        "median": value,
        "maximum": value,
    }


def _target_solve_receipt() -> dict[str, object]:
    chunks = []
    for index in range(7):
        chunks.append(
            {
                "chunk_index": index,
                "row_start": index * 8,
                "row_stop": (index + 1) * 8,
                "row_count": 8,
                "initial_kl": _summary(8, 1.0),
                "selected_kl": _summary(8, 0.5),
                "selected_step": _summary(8, 0.0),
                "trust_projection_count": _summary(8, 0.0),
                "token_locality": {
                    "method": (
                        "batched_projection_vs_same_rows_projected_as_"
                        "singletons"
                    ),
                    "probe_states": [
                        "native_teacher",
                        "a4_euclidean_baseline",
                    ],
                    "row_count": 8,
                    "nontrivial_multirow_probe": True,
                    "absolute_tolerance": micro._TOKEN_LOCALITY_ATOL,
                    "relative_tolerance": micro._TOKEN_LOCALITY_RTOL,
                    "teacher": {
                        "max_abs": 0.0,
                        "rms": 0.0,
                        "reference_max_abs": 0.0,
                    },
                    "a4_baseline": {
                        "max_abs": 0.0,
                        "rms": 0.0,
                        "reference_max_abs": 0.0,
                    },
                    "passed": True,
                },
                "full_solver_receipt_sha256": _digest(10 + index),
            }
        )
    target: dict[str, object] = {
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
        "initialization": (
            "float64_affine_sum_svd_pseudoinverse_minimum_norm"
        ),
        "canonical_target_dtype": "torch.float64",
        "affine_arithmetic_dtype": "torch.float64",
        "coordinate_layout": "joint_concatenated_four_node_rank_182",
        "runtime_correction_dtype": "torch.float32",
        "runtime_correction_cast_count_per_materialization": 1,
        "initial_correction_bit_identical_to_a4_float64_one_cast": True,
        "row_count": 56,
        "row_chunk_size": 8,
        "chunk_count": 7,
        "batching": {
            "one_batched_head_callback_per_optimizer_evaluation": True,
            "independent_adam_parameter_group_per_token": True,
            "independent_kl_best_checkpoint_per_token": True,
            "token_locality_audited_on_native_and_a4_states": True,
            "token_locality_absolute_tolerance": micro._TOKEN_LOCALITY_ATOL,
            "token_locality_relative_tolerance": micro._TOKEN_LOCALITY_RTOL,
        },
        "solver": {
            "steps": 64,
            (
                "learning_rate_fraction_of_per_token_"
                "initial_coefficient_rms"
            ): 1.0e-2,
            "minimum_scale_for_zero_rms": torch.finfo(torch.float64).eps,
            "initial_coefficient_rms": _summary(56, 1.0),
            "effective_learning_rate": _summary(56, 0.01),
            "scale_is_independent_for_each_token": True,
            "ridge": 0.0,
            "trust_radius": None,
            "initial_point_evaluated_as_safe_abstention": True,
        },
        "initial_kl": _summary(56, 1.0),
        "selected_kl": _summary(56, 0.5),
        "absolute_mean_kl_improvement": 0.5,
        "selected_not_worse_than_initial_for_every_token": True,
        "selected_step": _summary(56, 0.0),
        "trust_projection_count": _summary(56, 0.0),
        "initial_state_error": {
            "rmse": 1.0,
            "reference_rms": 2.0,
            "nrmse": 0.5,
            "max_abs_error": 2.0,
        },
        "selected_state_error": {
            "rmse": 0.5,
            "reference_rms": 2.0,
            "nrmse": 0.25,
            "max_abs_error": 1.0,
        },
        "hashes": {
            "initial_coefficient_sha256": _digest(1),
            "selected_coefficient_sha256": _digest(2),
            "initial_correction_sha256": _digest(3),
            "selected_correction_sha256": _digest(4),
            "initial_state_sha256": _digest(5),
            "selected_state_sha256": _digest(6),
        },
        "chunk_receipts": chunks,
        "frozen_affine_membership_by_construction": True,
        "basis_mean_or_decoder_changed": False,
        "deployable_generator_fitted": False,
        "contains_tensor_payloads": False,
    }
    target["receipt_sha256"] = micro._sha256_value(
        micro._TARGET_RECEIPT_DOMAIN,
        target,
    )
    return target


def _target_bridge_receipt(
    *,
    selected_coordinate_sha256: str,
) -> dict[str, object]:
    held_family = V8_FAMILY_LOFO_FAMILY_ALIASES[0]
    aliases = list(V8_FAMILY_LOFO_FAMILY_ALIASES[1:])
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    ranks = [46, 46, 45, 45]
    fragments = {
        name: f"fragment-{index}"
        for index, name in enumerate(nodes)
    }
    node_hashes = {
        name: _digest(20 + index) for index, name in enumerate(nodes)
    }
    coordinate_slices: dict[str, dict[str, int]] = {}
    start = 0
    for name, rank in zip(nodes, ranks, strict=True):
        coordinate_slices[name] = {
            "start": start,
            "stop": start + rank,
            "rank": rank,
        }
        start += rank

    family_masses = {alias: 1.0 / 7.0 for alias in aliases}
    role_fisher = {
        name: {
            "fragment_id": fragments[name],
            "total_mass": 1.0,
            "family_mass_by_alias": family_masses,
            "unit_total_mass": True,
            "equal_family_mass": True,
        }
        for name in nodes
    }

    def row_role(
        *,
        observations: int,
        sequences: int,
        seed: int,
    ) -> dict[str, object]:
        return {
            "row_key_sha256": _digest(seed),
            "observations": observations,
            "sequences": sequences,
            "fragment_tensor_sha256s": {
                fragment: {
                    "inputs_sha256": _digest(seed + 1 + index * 3),
                    "contributions_sha256": _digest(seed + 2 + index * 3),
                    "fisher_weights_sha256": _digest(seed + 3 + index * 3),
                }
                for index, fragment in enumerate(fragments.values())
            },
        }

    bridge: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_l10_l17_a5b_downstream_coordinate_targets"
        ),
        "format_version": 1,
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
            "held_family_alias": held_family,
            "held_family_rows_accepted": False,
            "held_family_used_for_fit_or_audit": False,
        },
        "authentication": {
            "compiled_inputs_sha256": _digest(40),
            "selected_joint_coordinates_sha256": (
                selected_coordinate_sha256
            ),
            "joint_coordinate_width": 182,
        },
        "frozen_affine_image": {
            "node_order": nodes,
            "rank_by_node": ranks,
            "coordinate_slices": coordinate_slices,
            "fragment_id_by_node": fragments,
            "basis_sha256_by_node": node_hashes,
            "mean_sha256_by_node": node_hashes,
            "encoder_sha256_by_node": node_hashes,
            "decoder_sha256_by_node": node_hashes,
            "basis_artifacts_unchanged_after_decode": True,
            "mean_tensors_byte_identical_after_decode": True,
            "encoder_tensors_byte_identical_after_decode": True,
            "decoder_tensors_byte_identical_after_decode": True,
        },
        "joint_roundtrip_audit": {
            "definition": (
                "summed_frozen_node_decodes_equal_joint_affine_decode"
            ),
            "joint_roundtrip_sha256": _digest(41),
            "summed_decoded_contribution_sha256": _digest(42),
            "max_abs_difference": 0.0,
            "rms_difference": 0.0,
            "relative_tolerance": 1.0e-11,
            "absolute_tolerance": 1.0e-11,
            "passed": True,
        },
        "inner_split": {
            "method": "deterministic_example_disjoint_hash_partition",
            "inner_split_binding_sha256": _digest(43),
            "inner_audit_examples_per_family": 1,
            "fit_example_count": 7,
            "audit_example_count": 7,
            "fit_example_membership_sha256": _digest(44),
            "audit_example_membership_sha256": _digest(45),
            "row_overlap_count": 0,
            "example_overlap_count": 0,
            "rows_exactly_partitioned": True,
            "examples_exactly_partitioned": True,
        },
        "fisher_normalization": {
            "all_rows_preserve_raw_authenticated_fisher_weights": True,
            "fit_and_audit_normalized_independently": True,
            "audit_weights_influence_fit_normalization": False,
            "policy": "equal_family_mass_within_each_role_and_node",
            "training_family_count": 7,
            "target_total_mass_per_role_and_node": 1.0,
            "target_mass_per_family": 1.0 / 7.0,
            "fit_by_node": role_fisher,
            "audit_by_node": copy.deepcopy(role_fisher),
        },
        "row_accounting": {
            "all": row_role(observations=56, sequences=14, seed=50),
            "fit": row_role(observations=28, sequences=7, seed=70),
            "audit": row_role(observations=28, sequences=7, seed=90),
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
    bridge["receipt_sha256"] = micro._sha256_value(
        micro._BRIDGE_RECEIPT_DOMAIN,
        bridge,
    )
    return bridge


@lru_cache(maxsize=1)
def _report() -> dict[str, object]:
    target = _target_solve_receipt()
    target_hashes = target["hashes"]
    assert isinstance(target_hashes, dict)
    bridge = _target_bridge_receipt(
        selected_coordinate_sha256=target_hashes[
            "selected_coefficient_sha256"
        ],
    )
    native_nll = 1.0
    conditions = {
        "layer10_only": _metric(
            native_nll=native_nll,
            delta_nll=0.30,
            kl=0.20,
            top1=0.50,
        ),
        "trajectory_corrected_layer17_only": _metric(
            native_nll=native_nll,
            delta_nll=0.20,
            kl=0.15,
            top1=0.60,
        ),
        "frozen_uncorrected_composition": _metric(
            native_nll=native_nll,
            delta_nll=0.50,
            kl=0.40,
            top1=0.40,
        ),
        "trajectory_corrected_composition": _metric(
            native_nll=native_nll,
            delta_nll=0.25,
            kl=0.20,
            top1=0.65,
        ),
        "matched_double_deletion": _metric(
            native_nll=native_nll,
            delta_nll=0.80,
            kl=0.60,
            top1=0.20,
        ),
    }
    node_names = ["node-a", "node-b", "node-c", "node-d"]
    node_hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(node_names)
    }
    generator_graph_sha256 = _digest(101)
    composition_graph_sha256 = _digest(102)
    bridge_sha256 = bridge["receipt_sha256"]
    accounting = bridge["row_accounting"]
    assert isinstance(bridge_sha256, str)
    assert isinstance(accounting, dict)
    fit_rows = accounting["fit"]
    audit_rows = accounting["audit"]
    assert isinstance(fit_rows, dict)
    assert isinstance(audit_rows, dict)
    fit_split_sha256 = micro._sha256_value(
        micro._FITTER_SPLIT_DOMAIN,
        {
            "role": "fit",
            "bridge_receipt_sha256": bridge_sha256,
            "row_key_sha256": fit_rows["row_key_sha256"],
            "observations": fit_rows["observations"],
        },
    )
    audit_split_sha256 = micro._sha256_value(
        micro._FITTER_SPLIT_DOMAIN,
        {
            "role": "audit",
            "bridge_receipt_sha256": bridge_sha256,
            "row_key_sha256": audit_rows["row_key_sha256"],
            "observations": audit_rows["observations"],
        },
    )
    freeze_payload = {
        "layer17_graph_sha256": generator_graph_sha256,
        "lowering_sha256_by_node": dict(node_hashes),
        "composition_graph_sha256": composition_graph_sha256,
        "bridge_receipt_sha256": bridge_sha256,
        "fit_split_sha256": fit_split_sha256,
        "audit_split_sha256": audit_split_sha256,
    }
    resources = {
        condition: {
            **lofo._EXPECTED_CONDITION_RESOURCES[condition],
            "executed_peak_live_modal_width": 0,
        }
        for condition in lofo._CONDITIONS
    }
    return micro.build_a5b_generator_microcanary_report(
        source_bindings=dict(micro._EXPECTED_SOURCE_BINDINGS),
        runtime={
            "model_id": "google/gemma-3-270m",
            "requested_revision": micro._EXPECTED_MODEL_REVISION,
            "model_fingerprint": micro._EXPECTED_MODEL_FINGERPRINT,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        capture={
            "capture_sha256": "a" * 64,
            "capture_audit_sha256": "b" * 64,
            "source_row_catalog_sha256": "c" * 64,
            "bounded_row_catalog_sha256": "d" * 64,
            "training_examples": 14,
            "captured_observations": 70,
            "bounded_target_rows": 56,
            "held_family_rows_present": False,
            "all_required_capture_audits_pass": True,
        },
        target_solve=target,
        target_bridge=bridge,
        generator_fit={
            "graph_sha256": generator_graph_sha256,
            "node_order": node_names,
            "parameter_count": 163_094,
            "macs_per_token": 160_352,
            "interaction_count": 0,
            "source_mean_bias_sha256_by_node": node_hashes,
            "source_decoder_basis_sha256_by_node": node_hashes,
            "lowering_sha256_by_node": node_hashes,
            "generator_plan_sha256_by_node": node_hashes,
        },
        frozen_executable={
            "corrected_layer17_graph_sha256": generator_graph_sha256,
            "corrected_layer17_lowering_sha256_by_node": node_hashes,
            "corrected_composition_graph_sha256": composition_graph_sha256,
            "bridge_receipt_sha256": bridge_sha256,
            "fit_split_sha256": fit_split_sha256,
            "audit_split_sha256": audit_split_sha256,
            "executable_freeze_sha256": micro._sha256_value(
                b"fisher-graph:a5b-executable-freeze:v1\0",
                freeze_payload,
            ),
            "corrected_layer17_parameters": 163_094,
            "corrected_layer17_macs_per_token": 160_352,
            "corrected_composition_parameters": 295_129,
            "corrected_composition_macs_per_token": 289_600,
            (
                "generator_and_composition_hashes_frozen_before_held_"
                "example_selection_or_model_evaluation"
            ): True,
        },
        evaluation={
            "execution_path": (
                "full_model_logits_fixed_capacity_a3_trajectory_lofo"
            ),
            "assessment_role": (
                "calibration_a_fit_family_blocked_development"
            ),
            "full_model_logits_scored": True,
            "full_model_compiled": False,
            "heldout_confirmation": False,
            "exact_resources_match_protocol": True,
            "latency_or_kernel_speed_claim": False,
            "supervised_tokens": 5,
            "logical_valid_tokens": 5,
            "native": {"nll_per_token": native_nll},
            "conditions": conditions,
            "resource_accounting": resources,
        },
    )


def _rehash(report: dict[str, object]) -> dict[str, object]:
    report.pop("report_sha256", None)
    report["report_sha256"] = micro._sha256_value(
        micro._REPORT_DOMAIN,
        report,
    )
    return report


def test_row_selector_preserves_capture_order_not_expected_example_order() -> None:
    row_keys = (
        ("example-b", 7),
        ("example-a", 9),
        ("example-b", 1),
        ("example-a", 3),
        ("example-a", 4),
        ("example-b", 2),
    )

    indices, selected = micro._select_first_rows_per_example(
        row_keys,
        expected_examples=("example-a", "example-b"),
        rows_per_example=2,
    )

    assert torch.equal(indices, torch.tensor((0, 1, 2, 3)))
    assert selected == row_keys[:4]


@pytest.mark.parametrize(
    ("row_keys", "expected_examples", "message"),
    (
        (
            (("example-a", 0), ("example-b", 0)),
            ("example-a", "example-c"),
            "capture row examples differ",
        ),
        (
            (
                ("example-a", 0),
                ("example-b", 0),
                ("example-a", 1),
            ),
            ("example-a", "example-b"),
            "too few captured rows",
        ),
    ),
)
def test_row_selector_rejects_missing_examples_or_rows(
    row_keys: tuple[tuple[str, int], ...],
    expected_examples: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        micro._select_first_rows_per_example(
            row_keys,
            expected_examples=expected_examples,
            rows_per_example=2,
        )


def test_report_roundtrip_is_strict_and_refuses_overwrite(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "microcanary.json"

    saved = micro.save_a5b_generator_microcanary_report(destination, report)
    loaded = micro.load_a5b_generator_microcanary_report(destination)

    assert saved == report
    assert loaded == report
    assert loaded["conclusion"] == report["conclusion"]
    with pytest.raises(FileExistsError, match="overwrite"):
        micro.save_a5b_generator_microcanary_report(destination, report)


def test_report_rejects_self_rehashed_contradictory_conclusion() -> None:
    tampered = copy.deepcopy(_report())
    conclusion = tampered["conclusion"]
    assert isinstance(conclusion, dict)
    conclusion["learned_generator_improves_frozen_kl"] = False

    with pytest.raises(ValueError, match="conclusion contradicts"):
        micro.validate_a5b_generator_microcanary_report(_rehash(tampered))


def test_report_rejects_self_rehashed_configuration_tamper() -> None:
    tampered = copy.deepcopy(_report())
    configuration = tampered["configuration"]
    assert isinstance(configuration, dict)
    configuration["generator_rank"] = 15

    with pytest.raises(ValueError, match="configuration drifted"):
        micro.validate_a5b_generator_microcanary_report(_rehash(tampered))


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("capture_leakage", "capture accounting drifted"),
        ("bridge_leakage", "outer split leaked the held family"),
        ("unsafe_prompt", "scope or safety contract drifted"),
    ),
)
def test_report_rejects_self_rehashed_leakage_and_safety_tampers(
    tamper: str,
    message: str,
) -> None:
    report = copy.deepcopy(_report())
    if tamper == "capture_leakage":
        capture = report["capture"]
        assert isinstance(capture, dict)
        capture["held_family_rows_present"] = True
    elif tamper == "bridge_leakage":
        bridge = report["target_bridge"]
        assert isinstance(bridge, dict)
        outer = bridge["outer_split"]
        assert isinstance(outer, dict)
        outer["held_family_rows_accepted"] = True
    elif tamper == "unsafe_prompt":
        safety = report["safety"]
        assert isinstance(safety, dict)
        safety["contains_prompt_text"] = True
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match=message):
        micro.validate_a5b_generator_microcanary_report(_rehash(report))
