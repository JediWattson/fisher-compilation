from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_cross_block_supermode_guard import (
    GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION,
    GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA,
    Gemma3CrossBlockSupermodeGuardThresholds,
    _SAFETY,
    _StratumTotals,
    _gate_one,
    _json_sha256,
    gemma3_cross_block_guard_stream_sha256,
    validate_gemma3_cross_block_supermode_guard_artifact,
)


def _batch() -> CalibrationBatch:
    valid = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid,
        },
        targets=torch.where(
            valid,
            input_ids,
            torch.full_like(input_ids, -100),
        ),
        valid_positions=valid,
        example_ids=("guard-a", "guard-b"),
    )


def _thresholds(stream_sha256: str) -> (
    Gemma3CrossBlockSupermodeGuardThresholds
):
    return Gemma3CrossBlockSupermodeGuardThresholds(
        source_model_fingerprint="1" * 64,
        source_execution_fingerprint="2" * 64,
        source_plan_artifact_sha256="3" * 64,
        source_replacement_oracle_artifact_sha256="4" * 64,
        guard_stream_sha256=stream_sha256,
        compiled_oracle_surface_maximum_absolute_error=0.0,
        compiled_oracle_behavior_maximum_absolute_difference=0.0,
        minimum_surface_recovery_fraction=1.0,
        maximum_absolute_delta_nll_per_token=0.0,
        maximum_teacher_kl_per_token=0.0,
        minimum_top1_agreement=1.0,
        minimum_nll_recovery_fraction=1.0,
        minimum_kl_recovery_fraction=1.0,
        minimum_top1_recovery_fraction=1.0,
    )


def test_stream_and_preregistered_threshold_hash_bind_labels() -> None:
    batch = _batch()
    families = {"guard-a": "family-a", "guard-b": "family-b"}
    lengths = {"guard-a": "short", "guard-b": "medium"}
    stream = gemma3_cross_block_guard_stream_sha256(
        (batch,),
        family_by_example=families,
        length_by_example=lengths,
    )
    second = gemma3_cross_block_guard_stream_sha256(
        (batch,),
        family_by_example=families,
        length_by_example=lengths,
    )
    changed = gemma3_cross_block_guard_stream_sha256(
        (batch,),
        family_by_example=families,
        length_by_example={"guard-a": "medium", "guard-b": "medium"},
    )

    assert stream == second
    assert stream != changed
    thresholds = _thresholds(stream)
    assert thresholds.metadata()["artifact_sha256"] == (
        thresholds.artifact_sha256
    )
    with pytest.raises(ValueError, match="exact savings"):
        Gemma3CrossBlockSupermodeGuardThresholds(
            **{
                **{
                    field: getattr(thresholds, field)
                    for field in thresholds.__dataclass_fields__
                },
                "expected_removed_parameters": 1279,
            }
        )


def _perfect_metrics() -> dict[str, object]:
    native_logits = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]]
    )
    ablation_logits = torch.tensor(
        [[[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]]
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    targets = torch.tensor([[0, 1]])
    native_surface = torch.ones(1, 2, 4)
    deleted_surface = torch.zeros_like(native_surface)
    native_surfaces = {
        "consumer_mlp_output": native_surface,
        "window_output": native_surface * 2,
        "final_logits": native_logits,
    }
    ablation_surfaces = {
        "consumer_mlp_output": deleted_surface,
        "window_output": deleted_surface,
        "final_logits": ablation_logits,
    }
    totals = _StratumTotals()
    totals.update(
        native_logits=native_logits,
        ablation_logits=ablation_logits,
        replacement_logits=native_logits,
        compiled_logits=native_logits,
        targets=targets,
        valid_positions=valid,
        native_surfaces=native_surfaces,
        ablation_surfaces=ablation_surfaces,
        replacement_surfaces=native_surfaces,
        compiled_surfaces=native_surfaces,
        ignore_index=-100,
    )
    return totals.finish()


def test_paired_metrics_recover_deletion_and_match_compiled_oracle() -> None:
    metrics = _perfect_metrics()
    recovery = metrics[
        "compiled_recovery_fraction_vs_coordinate_ablation"
    ]
    assert set(recovery["surfaces"].values()) == {1.0}
    assert set(recovery["behavior"].values()) == {1.0}
    exact = metrics["compiled_vs_coordinate_replacement"]
    assert all(
        surface["maximum_absolute_error"] == 0.0
        for surface in exact["surfaces"].values()
    )
    assert set(exact["behavior_absolute_differences"].values()) == {0.0}
    thresholds = _thresholds("5" * 64)
    assert all(_gate_one(metrics, thresholds).values())


def test_negative_recovery_threshold_disables_undefined_behavior_ratio() -> None:
    metrics = _perfect_metrics()
    recovery = metrics[
        "compiled_recovery_fraction_vs_coordinate_ablation"
    ]["behavior"]
    recovery["absolute_nll_displacement"] = None
    recovery["top1_disagreement"] = None
    thresholds = replace(
        _thresholds("6" * 64),
        minimum_nll_recovery_fraction=-1.0,
        minimum_top1_recovery_fraction=-1.0,
    )

    gates = _gate_one(metrics, thresholds)

    assert gates["compiled_recovers_absolute_nll_displacement"]
    assert gates["compiled_recovers_top1_disagreement"]


def _artifact() -> dict[str, object]:
    thresholds = _thresholds("7" * 64)
    metric = _perfect_metrics()
    metrics = {
        "overall": {"all": metric},
        "family": {"family-a": metric},
        "length": {"short": metric},
        "family_length": {'["family-a","short"]': metric},
    }
    gate_map = _gate_one(metric, thresholds)
    projection_names = (
        "anchor_gate",
        "anchor_up",
        "anchor_down",
        "consumer_gate",
        "consumer_up",
        "consumer_down",
    )
    call_counts = {
        condition: {
            origin: {
                name: (
                    1
                    if (
                        condition != "compiled" and origin == "source"
                    )
                    or (
                        condition == "compiled"
                        and origin == "candidate"
                    )
                    else 0
                )
                for name in projection_names
            }
            for origin in ("source", "candidate")
        }
        for condition in (
            "native",
            "coordinate_ablation",
            "coordinate_replacement",
            "compiled",
        )
    }
    payload = {
        "schema": GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA,
        "format_version": GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION,
        "binding": {
            "source_model_fingerprint": "1" * 64,
            "source_execution_fingerprint": "2" * 64,
            "source_plan_artifact_sha256": "3" * 64,
            "source_replacement_oracle_artifact_sha256": "4" * 64,
            "compiled_executor_fingerprint": "5" * 64,
            "threshold_artifact_sha256": thresholds.artifact_sha256,
            "guard_stream_sha256": "7" * 64,
        },
        "protocol": {
            "thresholds": thresholds.metadata(),
            "thresholds_fixed_before_guard": True,
            "candidate_frozen_before_guard": True,
            "native_ablation_replacement_compiled_paired": True,
            "coordinate_replacement_scale_frozen_from_oracle": True,
            "family_and_length_labels_frozen_in_stream_hash": True,
            "batch_count": 1,
            "example_count": 1,
            "family_counts": {"family-a": 1},
            "length_counts": {"short": 1},
            "ignore_index": -100,
        },
        "metrics": metrics,
        "gates": {
            "evaluated_strata": {
                "overall": {"all": gate_map},
                "family": {"family-a": gate_map},
                "length": {"short": gate_map},
            },
            "passed": True,
            "passing_does_not_authorize_any_next_split_or_execution": True,
        },
        "physical_execution_audit": {
            "projection_call_counts": call_counts,
            "expected_calls_per_projection_per_condition": 1,
            "compiled_source_consumer_gate_calls_zero": True,
            "compiled_source_consumer_up_calls_zero": True,
            "candidate_consumer_rows_physically_reduced_by_one": True,
            "all_projection_call_counts_exact": True,
            "candidate_consumer_gate_rows": 5,
            "source_consumer_gate_rows": 6,
            "candidate_consumer_up_rows": 5,
            "source_consumer_up_rows": 6,
            "full_consumer_down_projection_preserved": True,
        },
        "resource_accounting": {
            "source_whole_model_learned_parameters": 10_000,
            "candidate_whole_model_learned_parameters": 8_720,
            "removed_learned_parameters": 1280,
            "removed_linear_macs_per_valid_token": 1280,
            "carry_scale_macs_per_valid_token": 1,
            "net_arithmetic_macs_saved_per_valid_token": 1279,
            "guard_valid_position_count": 2,
            "removed_linear_macs_over_guard": 2_560,
            "carry_scale_macs_over_guard": 2,
            "net_arithmetic_macs_saved_over_guard": 2_558,
            "kernel_latency_speedup_claimed": False,
            "compression_materialized_in_evaluated_in_memory_overlay": True,
            "serialized_model_size_reduction_measured": False,
            "deployable_compressed_model_artifact_claimed": False,
        },
        "source_audit": {
            "guard_stream_unchanged": True,
            "source_model_state_unchanged": True,
            "source_execution_fingerprint_unchanged": True,
            "source_parameter_versions_unchanged": True,
            "source_parameter_gradients_absent": True,
            "compiled_executor_state_unchanged": True,
            "compiled_executor_parameter_versions_unchanged": True,
            "compiled_executor_parameter_gradients_absent": True,
            "source_modules_restored_after_overlay": True,
        },
        "safety": dict(_SAFETY),
    }
    return {**payload, "artifact_sha256": _json_sha256(payload)}


def test_strict_artifact_rejects_rehashed_resource_or_authority_tamper() -> None:
    artifact = _artifact()
    validate_gemma3_cross_block_supermode_guard_artifact(artifact)

    resource_tamper = copy.deepcopy(artifact)
    resource_tamper["resource_accounting"][
        "removed_learned_parameters"
    ] = 1279
    payload = dict(resource_tamper)
    payload.pop("artifact_sha256")
    resource_tamper["artifact_sha256"] = _json_sha256(payload)
    with pytest.raises(ValueError, match="resource accounting"):
        validate_gemma3_cross_block_supermode_guard_artifact(resource_tamper)

    authority_tamper = copy.deepcopy(artifact)
    authority_tamper["safety"]["authorizes_execution"] = True
    payload = dict(authority_tamper)
    payload.pop("artifact_sha256")
    authority_tamper["artifact_sha256"] = _json_sha256(payload)
    with pytest.raises(ValueError, match="authority"):
        validate_gemma3_cross_block_supermode_guard_artifact(authority_tamper)
