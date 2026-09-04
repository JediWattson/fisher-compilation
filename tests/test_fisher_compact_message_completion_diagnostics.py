from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fisher_graph import fisher_compact_message_completion_artifacts as artifacts
from fisher_graph.fisher_compact_message_completion_diagnostics import (
    DiagnosticContractError,
    build_full_rank_identity_smoke,
    calibrate_full_rank_identity,
    evaluate_compact_oracles,
    evaluate_direct_full_rank_identity_null,
    evaluate_m2_diagnostics,
    fit_mlp_codec_candidates,
    validate_four_block_capture,
    validate_full_rank_identity,
)
from fisher_graph.fisher_compact_message_completion_qualification import (
    CODEC_LADDER,
    SelectionGeometry,
    fit_selection_geometry,
    subset_partition_by_units,
)
from fisher_graph.fisher_compact_message_completion_science import (
    QContributionCodec,
    VFlowCodec,
)
from fisher_graph.fisher_compact_message_completion_synthetic import (
    SYNTHETIC_EVIDENCE_SCOPE,
    SyntheticCampaignData,
    SyntheticPartition,
    build_synthetic_campaign,
)


@pytest.fixture(scope="module")
def campaign() -> SyntheticCampaignData:
    return build_synthetic_campaign()


@pytest.fixture(scope="module")
def geometry(campaign: SyntheticCampaignData) -> SelectionGeometry:
    return fit_selection_geometry(campaign)


@pytest.fixture(scope="module")
def mlp_candidates(campaign: SyntheticCampaignData, geometry: SelectionGeometry):
    return fit_mlp_codec_candidates(campaign, geometry)


@pytest.fixture(scope="module")
def full_rank_smoke(
    campaign: SyntheticCampaignData, geometry: SelectionGeometry
):
    return build_full_rank_identity_smoke(campaign, geometry)


def _dependencies(name: str) -> dict[str, str]:
    return {
        dependency: "ab" * 32
        for dependency in artifacts.artifact_dependencies(name)
    }


def _build_artifact(name: str, body: dict[str, object]) -> dict[str, object]:
    return artifacts.build_artifact_payload(
        name,
        campaign_id="synthetic-diagnostic-test",
        dependency_hashes=_dependencies(name),
        body=body,
    )


def test_four_block_capture_uses_contiguous_floor_partition(
    campaign: SyntheticCampaignData,
) -> None:
    check = validate_four_block_capture(campaign.fit, campaign)
    width = campaign.source.mlp_gate.shape[0]
    assert check.block_boundaries == tuple(
        ((block * width) // 4, ((block + 1) * width) // 4)
        for block in range(4)
    )
    assert sum(end - start for start, end in check.block_boundaries) == width
    assert check.passed
    assert check.evidence_scope == SYNTHETIC_EVIDENCE_SCOPE


def test_corrupt_four_block_capture_fails_closed(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> None:
    tensors = dict(campaign.fit.tensors)
    corrupt = tensors["q_M_block"].copy()
    corrupt[:, 0, 0] += np.float32(0.1)
    tensors["q_M_block"] = corrupt
    bad_fit = replace(campaign.fit, tensors=tensors)
    bad_campaign = replace(campaign, fit=bad_fit)
    check = validate_four_block_capture(bad_fit, bad_campaign)
    assert not check.passed
    with pytest.raises(DiagnosticContractError, match="four-block MLP capture"):
        fit_mlp_codec_candidates(bad_campaign, geometry)


def test_mlp_codecs_are_fit_for_every_ladder_candidate_and_a_mid_chart(
    geometry: SelectionGeometry,
    mlp_candidates,
) -> None:
    assert [candidate.codec_id for candidate in mlp_candidates] == [
        item[0] for item in CODEC_LADDER
    ]
    for candidate, (_, q_rank, v_rank, ratio) in zip(
        mlp_candidates, CODEC_LADDER, strict=True
    ):
        assert candidate.q_rank == q_rank
        assert candidate.v_rank == v_rank
        assert candidate.regularization_ratio == ratio
        assert len(candidate.q_codecs) == geometry.mid_locator.chart_count
        assert len(candidate.v_codecs) == geometry.mid_locator.chart_count
        assert candidate.mid_locator_sha256 == geometry.mid_locator.identity_sha256
        assert all(codec.rank == q_rank for codec in candidate.q_codecs)
        assert all(codec.rank == v_rank for codec in candidate.v_codecs)
        assert all(codec.invariants.passed for codec in candidate.q_codecs)
        assert all(codec.invariants.passed for codec in candidate.v_codecs)
        assert candidate.fit_train_capture_validation.passed
        assert candidate.evidence_scope == SYNTHETIC_EVIDENCE_SCOPE


def test_mlp_codec_bundle_rejects_q_v_role_interchange(mlp_candidates) -> None:
    candidate = mlp_candidates[0]
    assert all(isinstance(codec, QContributionCodec) for codec in candidate.q_codecs)
    assert all(isinstance(codec, VFlowCodec) for codec in candidate.v_codecs)
    with pytest.raises(DiagnosticContractError, match="QContributionCodec"):
        replace(candidate, q_codecs=candidate.v_codecs)
    with pytest.raises(DiagnosticContractError, match="VFlowCodec"):
        replace(candidate, v_codecs=candidate.q_codecs)


def test_m2q_and_m2v_evaluate_both_independent_panels_and_validate_schema(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    mlp_candidates,
) -> None:
    result = evaluate_m2_diagnostics(campaign, geometry, mlp_candidates, "C2")
    assert [panel.partition_role for panel in result.m2q_panels] == [
        "fit_monitor",
        "tune",
    ]
    assert [panel.partition_role for panel in result.m2v_panels] == [
        "fit_monitor",
        "tune",
    ]
    assert result.evidence_scope == SYNTHETIC_EVIDENCE_SCOPE
    for ceiling in ("M2q", "M2v"):
        payload = _build_artifact(
            f"{ceiling.lower()}_results.json",
            result.artifact_body(ceiling),
        )
        assert payload["scientific_claim_eligible"] is False
        assert payload["protected_panel_accessed"] is False
        assert len(payload["body"]["panels"]) == 2


def test_m2_rejects_codec_bound_to_another_mid_locator(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    mlp_candidates,
) -> None:
    bad = replace(mlp_candidates[0], mid_locator_sha256="00" * 32)
    with pytest.raises(DiagnosticContractError, match="another A_mid locator"):
        evaluate_m2_diagnostics(campaign, geometry, (bad,), "C0")


def test_fixed_p32_compact_oracles_report_local_and_full_prompt_metrics(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> None:
    result = evaluate_compact_oracles(campaign, geometry, "C2")
    assert result.message_rank_per_head == 32
    assert result.evidence_scope == SYNTHETIC_EVIDENCE_SCOPE
    assert [panel.partition_role for panel in result.ambient_panels] == [
        "fit_monitor",
        "tune",
    ]
    assert [panel.partition_role for panel in result.chart_panels] == [
        "fit_monitor",
        "tune",
    ]
    for panel in result.ambient_panels + result.chart_panels:
        assert panel.local.q_total_max_abs_error >= 0.0
        assert panel.local.h_a_weighted_relative_l2 >= 0.0
        assert panel.local.h_plus_weighted_relative_l2 >= 0.0
        assert panel.full_prompt.metrics.supervised_token_count > 0
    ambient = _build_artifact(
        "compact_ambient_oracle.json",
        result.artifact_body("C-Ambient"),
    )
    chart = _build_artifact(
        "compact_chart_oracle.json",
        result.artifact_body("C-Chart"),
    )
    assert ambient["body"]["q_codec_projection"] is False
    assert chart["body"]["q_codec_projection"] is True
    assert ambient["body"]["selection_eligible"] is False
    assert chart["scientific_claim_eligible"] is False
    complete = result.as_protocol_dict("C-Ambient")
    assert complete["scientific_claim_eligible"] is False
    assert complete["guard_claim_eligible"] is False
    assert "local" in complete["panels"][0]
    assert "full_prompt_shadow" in complete["panels"][0]


def test_compact_oracle_refuses_a_non_p32_message_sketch(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> None:
    representations = tuple(
        replace(
            representation,
            basis=representation.basis[:, :16],
            coordinate_mean=representation.coordinate_mean[:16],
            coordinate_scale=representation.coordinate_scale[:16],
        )
        for representation in geometry.message_sketch.representations
    )
    bad_geometry = replace(
        geometry,
        message_sketch=replace(
            geometry.message_sketch,
            representations=representations,
        ),
    )
    with pytest.raises(DiagnosticContractError, match="frozen at p=32"):
        evaluate_compact_oracles(campaign, bad_geometry, "C2")


def test_full_rank_smoke_calibrates_all_rows_then_validates_monitor(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    full_rank_smoke,
) -> None:
    assert full_rank_smoke.passed
    calibration = full_rank_smoke.calibration_fit_train
    validation = full_rank_smoke.validation_fit_monitor
    assert calibration.passed
    assert validation.passed
    assert len(calibration.head_calibrations) == campaign.config.head_count
    expected_records = (
        len(CODEC_LADDER)
        * geometry.locator.chart_count
        * campaign.config.head_count
    )
    assert len(calibration.coefficient_calibrations) == expected_records
    assert len(validation.records) == expected_records
    assert calibration.normalization.message_centers.shape == (
        campaign.config.head_count,
        campaign.config.message_dimension,
    )
    assert calibration.normalization.coordinate_scale.shape == (
        campaign.config.head_count,
        campaign.config.message_dimension,
    )
    assert bool(
        (
            calibration.normalization.coordinate_scale
            >= torch.sqrt(torch.tensor(1e-6, dtype=torch.float64))
        ).all()
    )
    assert all(record.validation_row_count > 0 for record in validation.records)
    assert all(record.validation_analysis_unit_count > 0 for record in validation.records)
    assert all(record.zero_row_reason is None for record in validation.records)
    assert full_rank_smoke.evidence_scope == SYNTHETIC_EVIDENCE_SCOPE


def test_full_rank_tolerances_are_head_and_codec_chart_head_specific(
    geometry: SelectionGeometry,
    full_rank_smoke,
) -> None:
    calibration = full_rank_smoke.calibration_fit_train
    head_keys = {item.head_id for item in calibration.head_calibrations}
    assert head_keys == {0, 1, 2, 3}
    coefficient_keys = {
        (item.codec_id, item.chart_id, item.head_id)
        for item in calibration.coefficient_calibrations
    }
    assert coefficient_keys == {
        (candidate.codec_id, chart, head)
        for candidate in geometry.codec_candidates
        for chart in range(geometry.locator.chart_count)
        for head in range(4)
    }
    assert all(item.tau_full_c >= 1e-12 for item in calibration.coefficient_calibrations)
    assert all(
        item.tau_full_advantage == pytest.approx(10.0 * item.tau_full_c**2)
        for item in calibration.coefficient_calibrations
    )
    for record in full_rank_smoke.validation_fit_monitor.records:
        assert record.tau_full_m == calibration.head(record.head_id).tau_full_m
        assert record.tau_full_q == calibration.head(record.head_id).tau_full_q
        assert record.tau_full_c == calibration.coefficient(
            record.codec_id, record.chart_id, record.head_id
        ).tau_full_c


def test_full_rank_smoke_bodies_preserve_nonclaim_and_validate_schema(
    full_rank_smoke,
) -> None:
    artifact = _build_artifact(
        "full_rank_identity_smoke.json",
        full_rank_smoke.as_artifact_body(),
    )
    assert artifact["scientific_claim_eligible"] is False
    assert artifact["body"]["all_eligible_records_passed"] is True
    protocol = full_rank_smoke.as_protocol_dict()
    assert protocol["scientific_claim_eligible"] is False
    assert protocol["guard_claim_eligible"] is False
    assert "calibration_fit_train" in protocol
    assert "validation_fit_monitor" in protocol


def test_fit_monitor_zero_chart_support_fails_closed_with_reason(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> None:
    calibration = calibrate_full_rank_identity(campaign, geometry)
    one_unit = subset_partition_by_units(
        campaign.fit,
        (campaign.fit_split.left_units[0],),
    )
    validation = validate_full_rank_identity(
        one_unit,
        "fit_monitor",
        campaign,
        geometry,
        calibration,
    )
    assert not validation.passed
    assert validation.representation_candidate_status == (
        "invalid_zero_validation_support"
    )
    zero_records = [
        record
        for record in validation.records
        if record.status == "invalid_zero_validation_support"
    ]
    assert zero_records
    assert all(record.validation_row_count == 0 for record in zero_records)
    assert all(record.observed_max_error_c is None for record in zero_records)
    assert all(
        record.zero_row_reason
        == "no_valid_fit_monitor_rows_assigned_to_active_source_chart"
        for record in zero_records
    )


def test_missing_query_head_is_a_measurement_contract_failure(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> None:
    calibration = calibrate_full_rank_identity(campaign, geometry)
    fit_monitor = subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.left_units,
    )
    tensors = dict(fit_monitor.tensors)
    tensors["q_A_head"] = tensors["q_A_head"][:, :3, :]
    malformed = replace(fit_monitor, tensors=tensors)
    with pytest.raises(
        DiagnosticContractError,
        match="missing_head_in_validation_rows",
    ):
        validate_full_rank_identity(
            malformed,
            "fit_monitor",
            campaign,
            geometry,
            calibration,
        )


def test_direct_full_rank_null_reuses_frozen_smoke_without_refit(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    full_rank_smoke,
) -> None:
    result = evaluate_direct_full_rank_identity_null(
        campaign,
        geometry,
        full_rank_smoke,
        "C2",
    )
    assert result.passed
    assert result.representation_refit_performed is False
    assert result.frozen_tolerances_sha256 == (
        full_rank_smoke.frozen_tolerances_sha256
    )
    assert {record.codec_id for record in result.panel_validation.records} == {"C2"}
    assert result.message_roundtrip_passed
    assert result.q_roundtrip_passed
    assert result.coefficient_roundtrip_passed
    assert result.completion_target_null_passed
    assert result.coefficient_advantage_null_passed
    artifact = _build_artifact(
        "direct_full_rank_identity_null.json",
        result.as_artifact_body(),
    )
    assert artifact["scientific_claim_eligible"] is False
    assert artifact["body"]["downstream_vanishing_required"] is False


def test_direct_full_rank_null_refuses_a_failed_prerequisite_smoke(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    full_rank_smoke,
) -> None:
    failed_validation = replace(
        full_rank_smoke.validation_fit_monitor,
        representation_candidate_status="failed",
    )
    failed_smoke = replace(
        full_rank_smoke,
        validation_fit_monitor=failed_validation,
    )
    with pytest.raises(DiagnosticContractError, match="passing frozen P4 smoke"):
        evaluate_direct_full_rank_identity_null(
            campaign,
            geometry,
            failed_smoke,
            "C2",
        )


def test_diagnostic_fits_and_evaluations_repeat_exactly(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    mlp_candidates,
    full_rank_smoke,
) -> None:
    repeated_candidates = fit_mlp_codec_candidates(campaign, geometry)
    assert [item.identity_sha256 for item in repeated_candidates] == [
        item.identity_sha256 for item in mlp_candidates
    ]
    repeated_smoke = build_full_rank_identity_smoke(campaign, geometry)
    assert repeated_smoke.frozen_tolerances_sha256 == (
        full_rank_smoke.frozen_tolerances_sha256
    )
    first_m2 = evaluate_m2_diagnostics(campaign, geometry, mlp_candidates, "C2")
    second_m2 = evaluate_m2_diagnostics(
        campaign, geometry, repeated_candidates, "C2"
    )
    assert first_m2.m2q_panels == second_m2.m2q_panels
    assert first_m2.m2v_panels == second_m2.m2v_panels
    first_oracles = evaluate_compact_oracles(campaign, geometry, "C2")
    second_oracles = evaluate_compact_oracles(campaign, geometry, "C2")
    assert first_oracles.ambient_panels == second_oracles.ambient_panels
    assert first_oracles.chart_panels == second_oracles.chart_panels
