from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

from fisher_graph.fisher_compact_message_completion_qualification import (
    build_o1_qualification,
    assess_development_support,
    evaluate_o2,
    evaluate_o2q_replay,
    fidelity_gate_records,
    fit_selection_geometry,
    prepare_completion_panel,
    QualificationError,
    subset_partition_by_units,
)
from fisher_graph.fisher_compact_message_completion_science import (
    FisherCodec,
    QContributionCodec,
    VFlowCodec,
    evaluate_five_fidelity_gates,
)
from fisher_graph.fisher_compact_message_completion_synthetic import (
    build_synthetic_campaign,
)


def test_o1_calibrates_then_passes_disjoint_and_complete_fit() -> None:
    campaign = build_synthetic_campaign()
    result = build_o1_qualification(campaign)
    assert result.passed
    assert len(result.tolerances) == 6
    assert len(result.validation) == 6
    assert len(result.complete_fit) == 6
    assert all(item.tau_abs <= item.cap_abs for item in result.tolerances)
    assert all(item.tau_rel <= item.cap_rel for item in result.tolerances)
    assert all(item.calibration_passed for item in result.tolerances)
    assert all(item.passed for item in result.validation + result.complete_fit)


def test_selection_geometry_is_fit_train_only_and_satisfies_codec_invariants() -> None:
    campaign = build_synthetic_campaign()
    first = fit_selection_geometry(campaign)
    second = fit_selection_geometry(campaign)
    assert first.identity_sha256 == second.identity_sha256
    assert first.locator.chart_count == 4
    assert [
        first.locator.atlas_name,
        first.mid_locator.atlas_name,
        first.out_locator.atlas_name,
    ] == ["A_in", "A_mid", "A_out"]
    assert min(first.locator.fit_train_unit_counts) >= 8
    for locator in (first.locator, first.mid_locator, first.out_locator):
        assert len(locator.support_covariances) == locator.chart_count
        assert locator.sigma_perp_squared.shape == (locator.chart_count,)
        assert bool((locator.sigma_perp_squared >= 1e-12).all())
        assert bool(torch.isfinite(locator.parallel_support_limits).all())
        assert bool(torch.isfinite(locator.perpendicular_support_limits).all())
    assert first.message_sketch.rank == 32
    assert len(first.codec_candidates) == 4
    assert [(item.q_rank, item.v_rank, item.regularization_ratio) for item in first.codec_candidates] == [
        (32, 32, 1e-3),
        (64, 64, 1e-3),
        (32, 32, 1e-2),
        (64, 64, 1e-2),
    ]
    for candidate in first.codec_candidates:
        assert all(isinstance(codec, QContributionCodec) for codec in candidate.q_codecs)
        assert all(isinstance(codec, VFlowCodec) for codec in candidate.v_codecs)
        for codec in candidate.q_codecs + candidate.v_codecs:
            assert codec.invariants.err_orth < 1e-6
            assert codec.invariants.err_idem < 1e-6
            assert codec.invariants.err_encoder < 1e-10
            assert codec.invariants.passed


def test_codec_candidate_bundle_rejects_q_v_role_interchange() -> None:
    candidate = fit_selection_geometry(build_synthetic_campaign()).codec_candidates[0]
    with pytest.raises(QualificationError, match="QContributionCodec"):
        replace(candidate, q_codecs=candidate.v_codecs)
    with pytest.raises(QualificationError, match="VFlowCodec"):
        replace(candidate, v_codecs=candidate.q_codecs)
    source = candidate.q_codecs[0]
    untyped = FisherCodec(
        **{
            field.name: getattr(source, field.name)
            for field in fields(FisherCodec)
        }
    )
    with pytest.raises(QualificationError, match="QContributionCodec"):
        replace(candidate, q_codecs=(untyped,) + candidate.q_codecs[1:])


def test_locator_uses_local_distance_and_reports_frozen_support() -> None:
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    fit_train = subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.right_units,
        partition_role="fit",
    )
    assignments, supported = geometry.locator.assign_with_support(
        fit_train.tensors["h"]
    )
    assert torch.equal(assignments, geometry.locator.assign(fit_train.tensors["h"]))
    assert assignments.shape == supported.shape == (152,)
    assert set(int(value) for value in assignments) == {0, 1, 2, 3}
    extreme = torch.full((1, campaign.config.hidden_dimension), 1e6)
    _, extreme_supported = geometry.locator.assign_with_support(extreme)
    assert not bool(extreme_supported[0])


def test_all_synthetic_o2_candidates_pass_and_selection_is_deterministic() -> None:
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    evaluations, selection = evaluate_o2(campaign, geometry)
    assert selection.may_fit_j_models
    assert selection.selected_codec_id in {"C0", "C1", "C2", "C3"}
    assert set(selection.ranked_passing_codec_ids) == {"C0", "C1", "C2", "C3"}
    assert all(item.o2q_passed and item.o2v_passed for item in evaluations)
    for evaluation in evaluations:
        assert evaluate_five_fidelity_gates(evaluation.fit_monitor_q).passed
        assert evaluate_five_fidelity_gates(evaluation.tune_q).passed
        records = fidelity_gate_records(evaluation.tune_q)
        assert len(records) == 5
        assert all(record["passed"] for record in records)
        assert all(math.isfinite(float(record["observed"])) for record in records)


def test_fixed_completion_panel_keeps_all_four_heads_and_same_representation() -> None:
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    _, selection = evaluate_o2(campaign, geometry)
    assert selection.selected_codec_id is not None
    coefficient_dimension = geometry.codec(selection.selected_codec_id).q_rank
    panel = prepare_completion_panel(
        campaign.development_confirm,
        campaign,
        geometry,
        selection.selected_codec_id,
    )
    assert panel.chart_ids.shape == (96,)
    assert panel.active_chart_ids == (0, 1, 2, 3)
    assert panel.support_mask.shape == (96,)
    assert panel.support_mask.dtype == torch.bool
    assert panel.z.shape == (96, 32)
    assert panel.r.shape == (96, 4, 32)
    assert panel.c_base.shape == (96, 4, coefficient_dimension)
    assert panel.delta_c.shape == (96, 4, coefficient_dimension)
    assert panel.epsilon_completion.shape == (96, 4)
    assert panel.epsilon_j0.shape == (96, 4)
    assert bool(torch.isfinite(panel.delta_c).all())
    assert bool((panel.epsilon_completion > 0).all())
    assert float(panel.delta_c.square().mean()) > 0.0
    support = assess_development_support(panel)
    assert support.evaluable_analysis_unit_count == 24
    assert support.development_support_passed
    assert sum(count for _, count in support.chart_analysis_unit_counts) >= 24
    assert 0.0 <= support.out_of_support_rate <= 1.0
    unsupported = assess_development_support(
        replace(panel, support_mask=torch.zeros_like(panel.support_mask))
    )
    assert unsupported.out_of_support_rate == 1.0
    assert not unsupported.out_of_support_passed
    assert unsupported.evaluable_analysis_unit_count == 0
    assert not unsupported.aggregate_unit_support_passed
    assert unsupported.insufficient_independent_support
    replay = evaluate_o2q_replay(
        campaign.development_confirm,
        campaign,
        geometry,
        selection.selected_codec_id,
    )
    assert len(fidelity_gate_records(replay)) == 5


def test_nonmaterial_underpowered_chart_retains_aggregate_support() -> None:
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    _, selection = evaluate_o2(campaign, geometry)
    assert selection.selected_codec_id is not None
    panel = prepare_completion_panel(
        campaign.development_confirm, campaign, geometry, selection.selected_codec_id
    )
    units = tuple(
        sorted({key.analysis_unit_commitment for key in panel.row_keys})
    )

    def assessment_for_unit_count(unit_count: int):
        small = set(units[:unit_count])
        charts = torch.ones_like(panel.chart_ids)
        for index, key in enumerate(panel.row_keys):
            if key.analysis_unit_commitment in small:
                charts[index] = 0
        return assess_development_support(
            replace(
                panel,
                chart_ids=charts,
                active_chart_ids=(0, 1),
                support_mask=torch.ones_like(panel.support_mask),
            )
        )

    nonmaterial = assessment_for_unit_count(2)
    assert nonmaterial.underpowered_chart_ids == (0,)
    assert nonmaterial.materially_underpowered_chart_ids == ()
    assert nonmaterial.independent_support_passed

    material = assessment_for_unit_count(3)
    assert material.underpowered_chart_ids == (0,)
    assert material.materially_underpowered_chart_ids == (0,)
    assert not material.independent_support_passed


def test_coefficient_epsilons_are_fit_train_frozen_for_every_later_panel() -> None:
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    _, selection = evaluate_o2(campaign, geometry)
    assert selection.selected_codec_id is not None
    codec = geometry.codec(selection.selected_codec_id)
    fit_train = subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.right_units,
        partition_role="fit",
    )
    panels = (
        prepare_completion_panel(fit_train, campaign, geometry, codec.codec_id),
        prepare_completion_panel(campaign.tune, campaign, geometry, codec.codec_id),
        prepare_completion_panel(
            campaign.development_confirm, campaign, geometry, codec.codec_id
        ),
    )
    for panel in panels:
        assert torch.equal(
            panel.epsilon_completion,
            codec.epsilon_completion_by_chart_head[panel.chart_ids],
        )
        assert torch.equal(
            panel.epsilon_j0,
            codec.epsilon_j0_by_chart_head[panel.chart_ids],
        )
