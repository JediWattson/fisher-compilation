from __future__ import annotations

import copy
import hashlib
import json

import pytest

import fisher_graph.gemma3_l10_l17_full_block_closure_lofo as lofo
from fisher_graph.gemma3_l10_l17_full_block_closure_protocol import (
    build_default_gemma3_l10_l17_full_block_closure_protocol,
)


def _metric(
    native: float,
    delta: float,
    kl: float,
    top1: float,
) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _evaluation(
    index: int = 0,
    *,
    supervised_tokens: int = 100,
) -> dict[str, object]:
    native = 2.0 + index * 0.01
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
        "execution_path": "full_model_logits_a4_full_block_closure_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": supervised_tokens + 28,
        "native": {"nll_per_token": native},
        "conditions": {
            "layer10_only": _metric(native, 0.010, 0.010, 0.95),
            "source_layer17_only": _metric(native, 0.020, 0.020, 0.93),
            "a4_full_block_layer17_only": _metric(
                native, 0.015, 0.015, 0.94
            ),
            "frozen_uncorrected_composition": _metric(
                native, 0.050, 0.080, 0.85
            ),
            "a4_full_block_corrected_composition": _metric(
                native, 0.024, 0.040, 0.91
            ),
            "l10_edgeless_frozen_l17_composition": _metric(
                native, 0.048, 0.077, 0.86
            ),
            "l10_edgeless_a4_composition": _metric(
                native, 0.023, 0.039, 0.92
            ),
            "matched_double_deletion": _metric(
                native, 0.200, 0.200, 0.60
            ),
        },
        "resource_accounting": resources,
        "application_boundary": "layer.17.mlp.delta",
        "latency_or_kernel_speed_claim": False,
    }


def _capture_metadata() -> dict[str, object]:
    alignment = {
        "activation_fisher_and_activation_only_row_keys_equal": True,
        "native_and_compiled_row_keys_equal": True,
        "observations": 800,
        "sequences": 256,
        "fragment_count": 4,
        "row_key_sha256": "a" * 64,
    }
    trajectory_capture: dict[str, object] = {
        "alignment": {
            "observations": alignment["observations"],
            "sequences": alignment["sequences"],
            "fragment_count": alignment["fragment_count"],
            "row_key_sha256": alignment["row_key_sha256"],
        },
        "capture_sha256": "0" * 64,
    }
    trajectory_capture["capture_sha256"] = lofo._capture_metadata_sha256(
        trajectory_capture
    )
    tensor_names = {
        "native_post_attention_residual",
        "native_post_feedforward_delta",
        "native_block_output",
        "compiled_post_attention_residual",
        "compiled_post_feedforward_delta",
        "compiled_block_output",
        "exact_compact_retained_post_feedforward_delta",
        "algebraic_compact_retained_post_feedforward_delta",
        "a4_full_block_closure_target",
        "native_delta_only_closure",
        "residual_stream_closure_offset",
    }
    payload: dict[str, object] = {
        "schema": "synthetic-a4-capture",
        "format_version": 1,
        "scientific_role": (
            "paired_native_and_layer10_compiled_layer17_full_block_closure"
        ),
        "source_safe": True,
        "contains_tensors": False,
        "contains_prompt_text": False,
        "contains_prompt_identities": False,
        "contains_token_ids": False,
        "condition": "generated",
        "affected_layer_ordinals": [10],
        "layer_ordinal": 17,
        "trajectory_capture": trajectory_capture,
        "activation_only_capture": {
            "sites": {
                "post_attention_residual": "layer.17.post_attention",
                "post_feedforward_delta": "layer.17.mlp.delta",
                "block_output": "layer.17.output",
            },
            "pre_leaf_capture_uses_auxiliary_forward": True,
            "uses_autograd_grad": False,
            "post_attention_derived_from_block_output_subtraction": False,
        },
        "target": {
            "variant": "A4_full_block_closure",
            "application_boundary": "layer.17.mlp.delta",
            "uses_exact_compact_post_feedforward_delta": True,
            "uses_raw_compact_mlp_output": False,
            "formula": (
                "native_layer17_block_output-"
                "compiled_layer17_post_attention_residual-"
                "exact_compact_retained_layer17_post_feedforward_delta"
            ),
        },
        "alignment": alignment,
        "tensor_sha256s": {
            name: f"{position + 1:064x}"
            for position, name in enumerate(sorted(tensor_names))
        },
        "audits": {
            name: {
                "max_abs_difference": 0.0,
                "rms_difference": 0.0,
                "reference_rms": 1.0,
                "normalized_rms_difference": 0.0,
            }
            for name in (
                "native_block_decomposition",
                "compiled_block_decomposition",
                "a4_reconstruction",
                "a4_equivalent_formula",
                "a4_minus_delta_only_closure_offset_identity",
                "compact_post_feedforward_replay",
            )
        },
    }
    return {
        **payload,
        "capture_sha256": hashlib.sha256(
            lofo._A4_CAPTURE_DOMAIN + lofo._canonical_json_bytes(payload)
        ).hexdigest(),
    }


def _rehash_capture(metadata: dict[str, object]) -> None:
    payload = {
        key: value for key, value in metadata.items() if key != "capture_sha256"
    }
    metadata["capture_sha256"] = hashlib.sha256(
        lofo._A4_CAPTURE_DOMAIN + lofo._canonical_json_bytes(payload)
    ).hexdigest()


def test_aggregate_uses_token_weighted_micro_and_equal_family_macro() -> None:
    evaluations = [
        _evaluation(index, supervised_tokens=100 * (index + 1))
        for index in range(8)
    ]
    for index, evaluation in enumerate(evaluations):
        native = float(evaluation["native"]["nll_per_token"])
        delta = 0.01 * (index + 1)
        evaluation["conditions"]["a4_full_block_corrected_composition"] = (
            _metric(native, delta, 2.0 * delta, 0.90)
        )

    aggregate = lofo.aggregate_full_block_closure_folds(evaluations)
    micro = aggregate["micro"]["conditions"][
        "a4_full_block_corrected_composition"
    ]
    macro = aggregate["equal_family_macro"]["conditions"][
        "a4_full_block_corrected_composition"
    ]
    weighted_delta = sum(
        100 * (index + 1) * 0.01 * (index + 1) for index in range(8)
    ) / sum(100 * (index + 1) for index in range(8))

    assert aggregate["family_count"] == 8
    assert aggregate["completed_fold_count"] == 8
    assert micro["delta_nll_per_token"] == pytest.approx(weighted_delta)
    assert macro["delta_nll_per_token"] == pytest.approx(0.045)
    assert micro["top1_agreement_to_native"] == pytest.approx(0.90)


def test_aggregate_rejects_resource_drift_and_incomplete_fold_catalog() -> None:
    evaluations = [_evaluation(index) for index in range(8)]
    with pytest.raises(ValueError, match="exactly eight folds"):
        lofo.aggregate_full_block_closure_folds(evaluations[:-1])

    tampered = copy.deepcopy(evaluations)
    tampered[0]["resource_accounting"]["layer10_only"][
        "graph_parameters"
    ] += 1
    with pytest.raises(ValueError, match="resources drifted"):
        lofo.aggregate_full_block_closure_folds(tampered)


def test_evaluation_validation_survives_canonical_json_key_sorting() -> None:
    evaluations = [_evaluation(index) for index in range(8)]
    canonical = json.loads(lofo._canonical_json_bytes(evaluations))

    aggregate = lofo.aggregate_full_block_closure_folds(canonical)

    assert aggregate["completed_fold_count"] == 8


def test_a4_to_a3_gate_projection_preserves_only_base_gate_conditions() -> None:
    evaluation = _evaluation()

    projected = lofo._as_a3_gate_evaluation(evaluation)
    aggregate = lofo.aggregate_trajectory_correction_lofo_folds(
        [lofo._as_a3_gate_evaluation(_evaluation(index)) for index in range(8)]
    )

    assert tuple(projected["conditions"]) == (
        "layer10_only",
        "trajectory_corrected_layer17_only",
        "frozen_uncorrected_composition",
        "trajectory_corrected_composition",
        "matched_double_deletion",
    )
    assert projected["conditions"]["trajectory_corrected_composition"] == (
        evaluation["conditions"]["a4_full_block_corrected_composition"]
    )
    assert projected["resource_accounting"][
        "trajectory_corrected_layer17_only"
    ] == evaluation["resource_accounting"]["a4_full_block_layer17_only"]
    assert aggregate["family_count"] == 8


def test_full_block_gate_stack_passes_and_family_comparison_fails_closed() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    aliases = tuple(f"family-{index}" for index in range(8))
    evaluations = [_evaluation(index) for index in range(8)]
    aggregate = lofo.aggregate_full_block_closure_folds(evaluations)
    prior = {
        "equal_family_macro": {
            "native_to_candidate_kl_per_token": 0.050,
        },
        "native_to_candidate_kl_per_token_by_family_alias": {
            alias: 0.050 for alias in aliases
        },
    }
    common = {
        "protocol": protocol,
        "fold_aliases": aliases,
        "fold_evaluations": evaluations,
        "aggregate": aggregate,
        "prior_a3_comparison": prior,
        "exact_projection_metadata_match": True,
        "capture_audit": {"all_required_capture_audits_pass": True},
        "fold_bundle_receipt": {
            "fold_count": 8,
            "validated": True,
            "source_safe": True,
        },
        "source_model_unchanged": True,
        "layer10_unchanged": True,
    }

    decision = lofo.evaluate_full_block_closure_lofo_gates(**common)
    gates = {row["gate_id"]: row for row in decision["gate_table"]}

    assert decision["all_required_gates_pass"] is True
    assert gates["strict_family_macro_kl_improvement_vs_prior_a3"][
        "passed"
    ] is True
    assert gates["held_family_kl_improvement_count_vs_prior_a3"][
        "observed"
    ] == 8
    assert gates["post_feedforward_application_boundary"]["passed"] is True
    assert gates["full_block_capture_audits"]["passed"] is True
    assert gates["fold_executable_bundle"]["passed"] is True

    failed_prior = copy.deepcopy(prior)
    for alias in aliases[5:]:
        failed_prior["native_to_candidate_kl_per_token_by_family_alias"][
            alias
        ] = 0.030
    failed = lofo.evaluate_full_block_closure_lofo_gates(
        **{**common, "prior_a3_comparison": failed_prior}
    )
    failed_gates = {row["gate_id"]: row for row in failed["gate_table"]}

    assert failed_gates["held_family_kl_improvement_count_vs_prior_a3"][
        "observed"
    ] == 5
    assert failed_gates["held_family_kl_improvement_count_vs_prior_a3"][
        "passed"
    ] is False
    assert failed["all_required_gates_pass"] is False


def test_interaction_factorial_classifies_primary_and_non_explanations() -> None:
    primary = [_evaluation(index) for index in range(8)]
    for evaluation in primary:
        native = float(evaluation["native"]["nll_per_token"])
        evaluation["conditions"].update(
            {
                "l10_edgeless_frozen_l17_composition": _metric(
                    native, 0.03, 0.03, 0.90
                ),
                "frozen_uncorrected_composition": _metric(
                    native, 0.04, 0.04, 0.89
                ),
                "l10_edgeless_a4_composition": _metric(
                    native, 0.02, 0.02, 0.92
                ),
                "a4_full_block_corrected_composition": _metric(
                    native, 0.08, 0.08, 0.86
                ),
            }
        )

    primary_result = lofo._interaction_factorial(primary)
    nll = primary_result["metrics"]["delta_nll_per_token"]

    assert primary_result["classification"] == "primary_explanation"
    assert nll["equal_family_macro"]["difference_in_differences"] == (
        pytest.approx(0.05)
    )
    assert nll["difference_in_differences_fraction_of_gap"] == (
        pytest.approx(1.25)
    )
    assert primary_result["diagnostic_only"] is True
    assert primary_result["selected_or_mutated_topology"] is False

    non_explanatory = copy.deepcopy(primary)
    for evaluation in non_explanatory:
        native = float(evaluation["native"]["nll_per_token"])
        evaluation["conditions"][
            "a4_full_block_corrected_composition"
        ] = _metric(native, 0.03, 0.03, 0.91)

    assert lofo._interaction_factorial(non_explanatory)["classification"] == (
        "not_explanatory"
    )


def test_full_block_capture_audit_detects_hash_and_numerical_tampering() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    metadata = _capture_metadata()

    receipt = lofo._full_block_capture_audit_receipt(metadata, protocol)

    assert receipt["capture_hash_recomputed"] is True
    assert receipt["structural_contract_match"] is True
    assert receipt["all_required_capture_audits_pass"] is True

    hash_tampered = copy.deepcopy(metadata)
    hash_tampered["target"]["application_boundary"] = (
        "layer.17.mlp.operator_output"
    )
    hash_receipt = lofo._full_block_capture_audit_receipt(
        hash_tampered, protocol
    )
    assert hash_receipt["capture_hash_recomputed"] is False
    assert hash_receipt["structural_contract_match"] is False
    assert hash_receipt["all_required_capture_audits_pass"] is False

    numerical_tampered = copy.deepcopy(metadata)
    numerical_tampered["audits"]["a4_reconstruction"][
        "max_abs_difference"
    ] = 0.02
    _rehash_capture(numerical_tampered)
    numerical_receipt = lofo._full_block_capture_audit_receipt(
        numerical_tampered, protocol
    )
    assert numerical_receipt["capture_hash_recomputed"] is True
    assert numerical_receipt["structural_contract_match"] is True
    assert numerical_receipt["numerical_audits"]["a4_reconstruction"][
        "passed"
    ] is False
    assert numerical_receipt["all_required_capture_audits_pass"] is False
