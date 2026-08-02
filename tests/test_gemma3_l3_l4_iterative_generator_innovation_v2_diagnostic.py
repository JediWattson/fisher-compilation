from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_development as development,
)
from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_diagnostic as diagnostic,
)
from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_protocol as protocol,
)
from fisher_graph.gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
import test_gemma3_l3_l4_iterative_generator_innovation_v2_protocol as protocol_fixtures


def _metric(
    rmse: float,
    *,
    selected: str,
) -> dict[str, object]:
    return {
        "family_macro_rmse": rmse,
        "held_rmse_by_family": {
            f"family-{index}": rmse for index in range(8)
        },
        "selected_fit_candidate_id_by_family": {
            f"family-{index}": selected for index in range(8)
        },
    }


def _adaptive_metrics(
    *,
    scaled_feature: str = "ew16_scale_x1",
    full_feature: str = "ew04_scale_x1",
    scaled_active: bool = True,
    full_active: bool = True,
) -> dict[str, object]:
    scaled_fit = f"{scaled_feature}__conditional_ridge_10"
    full_fit = f"{full_feature}__conditional_ridge_10"
    folds = []
    for index in range(8):
        family = f"family-{index}"
        folds.append(
            {
                "held_family_id": family,
                "portfolio_selections": (
                    {
                        "portfolio_id": "scaled_l16",
                        "selection": {
                            "selected_fit_candidate_id": scaled_fit,
                        },
                    },
                    {
                        "portfolio_id": "full_temporal_grid",
                        "selection": {
                            "selected_fit_candidate_id": full_fit,
                        },
                    },
                ),
                "outer_candidate_receipts": (
                    {
                        "fit_candidate_id": scaled_fit,
                        "feature_candidate_id": scaled_feature,
                        "fit": {"conditional_active": scaled_active},
                    },
                    {
                        "fit_candidate_id": full_fit,
                        "feature_candidate_id": full_feature,
                        "fit": {"conditional_active": full_active},
                    },
                ),
            }
        )
    return {
        "folds": tuple(folds),
        "metrics": {
            "family_macro_parent_rmse": 1.10,
            "held_parent_rmse_by_family": {
                f"family-{index}": 1.10 for index in range(8)
            },
            "v1_structural_gate_macro_not_worse_than_legacy_shared": False,
            "v1_metrics": _metric(
                0.96,
                selected=(
                    "exact_v1_ew16_tau1__conditional_ridge_10"
                ),
            ),
            "static_u_metrics": _metric(1.0, selected="static_u"),
            "legacy_shared_metrics": _metric(
                0.95,
                selected="legacy_shared_q6_first_two",
            ),
            "portfolio_metrics": {
                "scaled_l16": _metric(
                    0.80,
                    selected=scaled_fit,
                ),
                "current_only": _metric(
                    0.90,
                    selected=(
                        "current_only_scale_x1__conditional_ridge_10"
                    ),
                ),
                "full_temporal_grid": _metric(
                    0.60,
                    selected=full_fit,
                ),
            },
        }
    }


def test_scale_two_stage_health_freeze_cannot_change_candidate_specs() -> None:
    final_summaries = protocol_fixtures._summaries()
    provisional_summaries = copy.deepcopy(final_summaries)
    for example in provisional_summaries.values():
        for row in example["candidate_health_by_id"].values():
            row["q90_absolute_bounded_by_channel"] = (0.0, 0.0)
            row["central_fraction_by_channel"] = (0.0, 0.0)
            row["bounded_trace_sha256"] = "0" * 64
            row["candidate_feature_receipt_sha256"] = "0" * 64
    provisional = (
        protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
            per_example_raw_summaries=provisional_summaries,
            prior_lineage=protocol_fixtures._lineage(),
        )
    )
    final = protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=final_summaries,
        prior_lineage=protocol_fixtures._lineage(),
    )
    assert protocol.generator_innovation_v2_candidate_specs(
        provisional
    ) == protocol.generator_innovation_v2_candidate_specs(final)


def test_target_candidate_records_use_shared_analyzer_r4_coordinates() -> None:
    scores = torch.tensor(
        (
            (1.0, 0.0, 0.5, -0.25),
            (0.0, 1.0, -0.5, 0.25),
        ),
        dtype=torch.float64,
    )
    target = torch.tensor((0.1, -0.2), dtype=torch.float64)
    record = diagnostic._build_candidate_prompt_record(
        example_id="example",
        family_id="family",
        token_scores=scores,
        compensation_target=target,
    )
    assert record.coordinate_names == GENERATOR_INNOVATION_TANGENT_ORDER


def test_target_rejects_scale_wrapper_hash_in_place_of_standalone_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_gemma_iterative_generator_innovation_v2_candidate_plan",
        lambda _value: None,
    )
    monkeypatch.setattr(
        development,
        (
            "validate_gemma_iterative_generator_innovation_v2_"
            "scale_development_report"
        ),
        lambda _value: None,
    )
    standalone = "a" * 64
    wrapper = "b" * 64
    with pytest.raises(ValueError, match="standalone scale receipt"):
        build = getattr(
            development,
            (
                "build_gemma_iterative_generator_innovation_v2_"
                "target_development_report"
            ),
        )
        build(
            record_bank={},
            legacy_records=(),
            fixed_basis=(),
            candidate_plan={
                "lineage": {
                    "v2_scale_receipt_sha256": "c" * 64,
                    "v2_scale_receipt_file_sha256": standalone,
                }
            },
            candidate_plan_file_sha256="d" * 64,
            scale_receipt_file_sha256=wrapper,
            scale_development_report={
                "scale_receipt": {"receipt_sha256": "c" * 64},
                "lineage": {"scale_receipt_file_sha256": standalone},
            },
            scale_development_report_file_sha256=wrapper,
            live_lineage={},
            candidate_feature_receipt_sha256_by_example_id={},
            raw_trace_receipt_sha256_by_example_id={},
            score_receipt_sha256_by_example_id={},
            token_vjp_artifact_sha256_by_example_id={},
            total_backward_call_count=0,
            vjp_chunk_size=8,
            source_code_sha256_by_file={},
        )


def test_backward_call_receipt_is_exact_sum_of_per_prompt_chunks() -> None:
    adaptive = {
        "legacy_prompt_fisher_records": (
            {"supervised_tokens": 1},
            {"supervised_tokens": 8},
            {"supervised_tokens": 9},
            {"supervised_tokens": 17},
        )
    }
    assert development._expected_vjp_backward_calls(
        adaptive,
        vjp_chunk_size=8,
    ) == 1 + 1 + 2 + 3


def test_attribution_allows_real_scale_rescue_when_v1_loses_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_generator_innovation_adaptive_v2_report",
        lambda _value: None,
    )
    decision = development._attribution_decision(
        adaptive_report=_adaptive_metrics(),
        gate_config=development._FROZEN_ATTRIBUTION_GATES,
        base_gate_config=development.GENERATOR_INNOVATION_GATE_CONFIG,
    )
    assert decision["gates"] == {
        "scale_rescue_passed": True,
        "memory_rescue_passed": True,
        "temporal_value_passed": True,
    }
    assert decision["nomination"]["nominated_recipe_id"] == (
        "full_temporal_grid"
    )
    assert (
        decision["nomination"]["finite_displacement_or_provider_authorized"]
        is False
    )


def test_attribution_requires_non_l16_selection_for_memory_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_generator_innovation_adaptive_v2_report",
        lambda _value: None,
    )
    adaptive = _adaptive_metrics(
        full_feature="ew16_scale_x1",
    )
    decision = development._attribution_decision(
        adaptive_report=adaptive,
        gate_config=development._FROZEN_ATTRIBUTION_GATES,
        base_gate_config=development.GENERATOR_INNOVATION_GATE_CONFIG,
    )
    assert decision["gates"]["memory_rescue_passed"] is False
    assert decision["nomination"]["nominated_recipe_id"] == "scaled_l16"


def test_attribution_rejects_rehashed_widened_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_generator_innovation_adaptive_v2_report",
        lambda _value: None,
    )
    widened = dict(development._FROZEN_ATTRIBUTION_GATES)
    widened["minimum_material_family_win_count"] = 0
    with pytest.raises(ValueError, match="widened"):
        development._attribution_decision(
            adaptive_report=_adaptive_metrics(),
            gate_config=widened,
            base_gate_config=development.GENERATOR_INNOVATION_GATE_CONFIG,
        )


def test_attribution_reads_conditional_activity_from_selected_outer_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_generator_innovation_adaptive_v2_report",
        lambda _value: None,
    )
    decision = development._attribution_decision(
        adaptive_report=_adaptive_metrics(
            scaled_active=False,
            full_active=False,
        ),
        gate_config=development._FROZEN_ATTRIBUTION_GATES,
        base_gate_config=development.GENERATOR_INNOVATION_GATE_CONFIG,
    )
    assert decision["selection_counts"][
        "scaled_l16_active_conditional_fold_count"
    ] == 0
    assert decision["selection_counts"][
        "full_temporal_active_conditional_fold_count"
    ] == 0
    assert decision["gates"]["scale_rescue_passed"] is False
    assert decision["gates"]["memory_rescue_passed"] is False


def test_full_nomination_does_not_require_scaled_l16_viability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_generator_innovation_adaptive_v2_report",
        lambda _value: None,
    )
    adaptive = _adaptive_metrics()
    adaptive["metrics"]["portfolio_metrics"]["scaled_l16"] = _metric(
        1.05,
        selected="ew16_scale_x1__conditional_ridge_10",
    )
    decision = development._attribution_decision(
        adaptive_report=adaptive,
        gate_config=development._FROZEN_ATTRIBUTION_GATES,
        base_gate_config=development.GENERATOR_INNOVATION_GATE_CONFIG,
    )
    assert decision["gates"]["scale_rescue_passed"] is False
    assert decision["own_viability"]["full_temporal_grid"][
        "performance_gate_passed"
    ] is True
    assert decision["nomination"]["nominated_recipe_id"] == (
        "full_temporal_grid"
    )
    assert decision["own_viability"]["full_temporal_grid"][
        "complete_base_gate_set_passed"
    ] is None
