import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fisher_graph.research_figures import (
    extract_research_figure_data,
    render_bilinear_spectral_assessment,
    render_l3_l4_rank_diagnostic,
    render_reference_provider_collision_attenuation,
    render_reference_provider_v3_assessment,
    render_research_ladder,
    verify_available_source_digests,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "research"
    / "current_research_summary_v1.json"
)
LADDER_PATH = REPOSITORY_ROOT / "docs" / "images" / "research-ladder.svg"
DIAGNOSTIC_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "l3-l4-rank-diagnostic.svg"
)
BILINEAR_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "bilinear-spectral-assessment.svg"
)
ATTENUATION_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "reference-provider-collision-attenuation.svg"
)
V3_ASSESSMENT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "reference-provider-v3-assessment.svg"
)


def _load_summary() -> tuple[bytes, dict[str, object]]:
    summary_bytes = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    assert isinstance(summary, dict)
    return summary_bytes, summary


def _render_expected() -> tuple[str, str, str, str, str, str]:
    summary_bytes, summary = _load_summary()
    source_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    data = extract_research_figure_data(summary)
    source_label = SUMMARY_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    return (
        render_research_ladder(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        render_l3_l4_rank_diagnostic(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        render_bilinear_spectral_assessment(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        render_reference_provider_collision_attenuation(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        render_reference_provider_v3_assessment(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        source_sha256,
    )


def test_research_summary_data_contract() -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)

    assert [stage.status for stage in data.stages] == [
        "verified_reference",
        "fidelity_parent",
        "open_development",
        "analysis_only",
        "analysis_only",
        "frozen_assessment",
        "sealed_mixed_result",
    ]
    assert data.stages[0].resource == (
        "35.3% of estimated source-block multiplies"
    )
    assert data.stages[0].fidelity == (
        "Exact validation argmax; test result is exploratory"
    )
    assert len(data.sources) == 13
    assert data.sources[3].sha256 == (
        "4a6e2437711f77af0123fd8fd3c8f35bb557f36623da6ef3272bb7f665ddd016"
    )
    assert data.sources[-8].sha256 == (
        "ea42a293e4d5f4c1a6ef68b0a60826a14bc61b0e5e8ac373171d4a331d43d671"
    )
    assert data.sources[-7].sha256 == (
        "931596c3889fe80c822c8620ca2ea9351751a98e93c3a49f4edce1713650ef3d"
    )
    assert data.sources[-6].sha256 == (
        "856d116f687fcde936e447d8f14053e74fa9ebf3a6996a60c527cec2e541a37a"
    )
    assert data.sources[-5].sha256 == (
        "6963ba73b71d178e66c58bbcdaf9d1ca9feffb51ce1ad062599b55bdd3f753ab"
    )
    assert data.sources[-4].sha256 == (
        "1e14518f915821aa7448b6f4799e322e2451074b3030ba4107c6a2a0924be4d9"
    )
    assert data.sources[-3].sha256 == (
        "613856ec39a7d0cac21cc6e41a155a4609c73ea05e4daa01ccf1affe26153b6e"
    )
    assert data.sources[-2].sha256 == (
        "e6af80a6929b79fb86a4fabb2b0bf94cea92a881d37506802091bf2ddb0e804a"
    )
    assert data.sources[-1].sha256 == (
        "df4562f976ae903fc89d6d299b4cb3fbd771f99b28e28717d545d9fdb48f0392"
    )
    assert data.reference_provider.selected_candidate_id == (
        "spectral-r08-t08"
    )
    assert (
        data.reference_provider.selected_stored_scalar_count,
        data.reference_provider.dense_provider_stored_scalar_count,
    ) == (910, 15_046)
    assert data.reference_provider.provider_scalar_reduction_fraction == (
        pytest.approx(0.939518808985777)
    )
    assert (
        data.reference_provider.assessment_fidelity_and_structure_gates_passed
        == 11
    )
    assert not (
        data.reference_provider.assessment_collision_panel_gate_passed
    )
    assert (
        data.reference_provider.assessment_minimum_collision_target_relative_difference
        < data.reference_provider.assessment_collision_threshold
    )
    attenuation = data.collision_attenuation
    assert attenuation.collision_endpoint_count == 40
    assert attenuation.all_target_hashes_match_opened_assessment
    assert (
        attenuation.unordered_pair_count,
        attenuation.gate_witness_count,
        attenuation.numerically_valid_gate_witness_count,
    ) == (32, 16, 16)
    assert (
        attenuation.gate_witnesses_at_or_above_threshold,
        attenuation.gate_witnesses_below_threshold,
    ) == (4, 12)
    assert (
        attenuation.reference_baseline_relative_dilution_count,
        attenuation.pre_ff_norm_attenuation_count,
    ) == (10, 4)
    assert (
        attenuation.minimum_failed_witness_retained_64_fisher_energy_fraction
        == pytest.approx(0.9999880706223239)
    )
    assert not attenuation.formal_v2_decision_changed
    assert attenuation.fresh_v3_assessment_required
    v3 = data.reference_provider_v3
    assert (
        v3.assessment_probe_count,
        v3.ordinary_fidelity_probe_count,
        v3.contrast_probe_count,
        v3.contrast_pair_count,
    ) == (48, 16, 32, 24)
    assert v3.ordinary_fidelity_passed
    assert v3.ordinary_fidelity_fisher_weighted_relative_error == pytest.approx(
        0.06772962100197875
    )
    assert v3.ordinary_fidelity_reference_cosine == pytest.approx(
        0.9977221137523479
    )
    radial, signed, intended_null = v3.contrast_families
    assert (
        radial.teacher_qualified_contrast_count,
        radial.candidate_pass_count,
    ) == (8, 0)
    assert radial.worst_contrast_relative_error == pytest.approx(
        1.302609636349226
    )
    assert (
        signed.teacher_qualified_contrast_count,
        signed.candidate_pass_count,
    ) == (1, 0)
    assert signed.minimum_direction_cosine == pytest.approx(
        -0.9215394885190904
    )
    assert signed.minimum_projection_gain == pytest.approx(
        -2.4427641222529104
    )
    assert (
        intended_null.teacher_qualified_contrast_count,
        intended_null.candidate_pass_count,
    ) == (12, 7)
    assert v3.formal_outcome == "panel_inconclusive_sensitivity"
    assert not v3.provider_passed

    rank_64, rank_128 = data.diagnostic.rank_results
    assert (rank_64.rank, rank_128.rank) == (64, 128)
    assert (
        rank_64.source_reconstruction_relative_l2,
        rank_128.source_reconstruction_relative_l2,
    ) == pytest.approx((0.2911029334366325, 0.1253992516767284))
    assert (
        rank_64.target_reconstruction_relative_l2,
        rank_128.target_reconstruction_relative_l2,
    ) == pytest.approx((0.2290740597055826, 0.1311348111849064))
    assert (
        rank_64.in_sample_jvp_relative_residual,
        rank_128.in_sample_jvp_relative_residual,
    ) == pytest.approx((0.3226633141461419, 0.2573207079237517))
    assert (
        rank_64.pair_output_cosine,
        rank_128.pair_output_cosine,
    ) == pytest.approx((0.7633875906308878, 0.5995991389752001))
    assert (
        rank_64.pair_output_relative_l2,
        rank_128.pair_output_relative_l2,
    ) == pytest.approx((1.1867917924590743, 1.7710422189550394))
    assert (
        rank_64.pair_parameter_fraction_of_flat,
        rank_128.pair_parameter_fraction_of_flat,
    ) == pytest.approx((0.11397345823575332, 0.25136612021857924))
    assert (
        rank_64.whole_model_parameter_fraction_of_source,
        rank_128.whole_model_parameter_fraction_of_source,
    ) == pytest.approx((0.9931496390433154, 0.994211897658625))
    assert data.diagnostic.content_disjoint
    assert not data.diagnostic.family_disjoint
    assert not data.diagnostic.reference_provider_compiled
    assert (
        data.bilinear.selected_source_rank,
        data.bilinear.selected_target_rank,
    ) == (8, 8)
    assert (
        data.bilinear.selected_plan_stored_coefficient_count,
        data.bilinear.direct_dense_branch_coefficient_count,
    ) == (6880, 172032)
    assert (
        data.bilinear.base_plus_branch_stored_coefficient_count,
        data.bilinear.matched_dense_three_branch_coefficient_count,
    ) == (46816, 958464)
    assert data.bilinear.branch_coefficient_reduction_fraction == pytest.approx(
        0.9600074404761905
    )
    assert data.bilinear.combined_coefficient_reduction_fraction == pytest.approx(
        0.9511551816239316
    )
    assert (
        data.bilinear.selection_base_relative_error,
        data.bilinear.selection_augmented_relative_error,
        data.bilinear.selection_error_reduction_fraction,
        data.bilinear.selection_augmented_cosine,
    ) == pytest.approx(
        (
            0.2072601933625187,
            0.1685161246715486,
            0.18693444246287494,
            0.9872879962098491,
        )
    )
    assert (
        data.bilinear.assessment_base_relative_error,
        data.bilinear.assessment_augmented_relative_error,
        data.bilinear.assessment_error_reduction_fraction,
        data.bilinear.assessment_augmented_cosine,
        data.bilinear.assessment_c11_relative_error,
        data.bilinear.assessment_c11_cosine,
    ) == pytest.approx(
        (
            0.20900929122427236,
            0.1693739874258058,
            0.1896341716021459,
            0.9871028249581452,
            0.22976163514361414,
            0.9740618920255555,
        )
    )
    assert data.bilinear.decision == "passes_frozen_assessment"
    assert not data.bilinear.prompt_conditioned_reference_provider_compiled
    assert not data.bilinear.nll_measured
    assert not data.bilinear.model_parameter_compression_claim
    assert not data.bilinear.latency_measured


@pytest.mark.parametrize(
    ("path", "figure_index"),
    [
        (LADDER_PATH, 0),
        (DIAGNOSTIC_PATH, 1),
        (BILINEAR_PATH, 2),
        (ATTENUATION_PATH, 3),
        (V3_ASSESSMENT_PATH, 4),
    ],
)
def test_committed_research_figure_matches_summary(
    path: Path,
    figure_index: int,
) -> None:
    (
        expected_ladder,
        expected_diagnostic,
        expected_bilinear,
        expected_attenuation,
        expected_v3_assessment,
        source_sha256,
    ) = _render_expected()
    expected = (
        expected_ladder,
        expected_diagnostic,
        expected_bilinear,
        expected_attenuation,
        expected_v3_assessment,
    )[figure_index]

    assert path.read_text(encoding="utf-8") == expected
    root = ET.fromstring(expected)
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{namespace}svg"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"] == (
        "figure-title figure-description"
    )
    title = root.find(f"{namespace}title")
    description = root.find(f"{namespace}desc")
    metadata = root.find(f"{namespace}metadata")
    assert title is not None and title.text
    assert description is not None and description.text
    assert metadata is not None
    assert source_sha256 in (metadata.text or "")
    assert "artifacts/research/current_research_summary_v1.json" in (
        metadata.text or ""
    )
    assert "gemma_l3_l4_rank_64:" in (metadata.text or "")
    assert "gemma_bilinear_assessment:" in (metadata.text or "")
    assert "gemma_reference_provider_v2_assessment:" in (
        metadata.text or ""
    )
    assert "gemma_reference_provider_v2_attenuation_localization:" in (
        metadata.text or ""
    )
    assert "gemma_reference_provider_v3_assessment:" in (
        metadata.text or ""
    )
    assert "@media (prefers-color-scheme: dark)" in expected


def test_research_summary_rejects_unknown_format() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["format_version"] = 999

    with pytest.raises(ValueError, match="format_version must be 1"):
        extract_research_figure_data(summary)


def test_research_summary_rejects_unknown_stage_status() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["research_ladder"][3]["status"] = "compression_proven"

    with pytest.raises(ValueError, match="unsupported research stage status"):
        extract_research_figure_data(summary)


def test_available_upstream_source_digests_are_verified() -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)

    verified = verify_available_source_digests(
        data.sources,
        source_root=REPOSITORY_ROOT,
    )
    expected_available = {
        source.source_id
        for source in data.sources
        if (REPOSITORY_ROOT / source.path).is_file()
    }

    assert "toy_fused_executor" in verified
    assert set(verified) == expected_available


def test_upstream_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)
    source = data.sources[0]
    destination = tmp_path / source.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"not the authenticated report")

    with pytest.raises(ValueError, match="research source digest mismatch"):
        verify_available_source_digests(
            (source,),
            source_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "family_disjoint",
            True,
            "rank diagnostic contract requires content-disjoint",
        ),
        (
            "reference_provider_compiled",
            True,
            "rank diagnostic contract requires content-disjoint",
        ),
        (
            "logical_lags",
            [0, 4],
            "logical_lags must be contiguous 0 through 4",
        ),
    ],
)
def test_fixed_rank_diagnostic_protocol_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["l3_l4_diagnostic"][field] = value

    with pytest.raises(ValueError, match=message):
        extract_research_figure_data(summary)


def test_rank_diagnostic_rejects_reversed_finding() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["l3_l4_diagnostic"]["rank_results"][1][
        "pair_output_relative_l2"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="requires worsening pair-output relative L2",
    ):
        extract_research_figure_data(summary)


def test_rank_diagnostic_supports_signed_cosine() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    rank_results = summary["l3_l4_diagnostic"]["rank_results"]
    rank_results[0]["pair_output_cosine"] = -0.1
    rank_results[1]["pair_output_cosine"] = -0.2

    data = extract_research_figure_data(summary)

    assert [row.pair_output_cosine for row in data.diagnostic.rank_results] == [
        -0.1,
        -0.2,
    ]


def test_bilinear_diagnostic_rejects_inconsistent_accounting() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["bilinear_diagnostic"][
        "branch_coefficient_reduction_fraction"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="inconsistent branch coefficient reduction",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_conditioned_reference_provider_compiled",
        "nll_measured",
        "model_parameter_compression_claim",
        "latency_measured",
        "positive_pair_identity_generalization_claim",
    ],
)
def test_bilinear_diagnostic_preserves_claim_boundaries(
    field: str,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["bilinear_diagnostic"][field] = True

    with pytest.raises(
        ValueError,
        match="must preserve provider, NLL, model compression",
    ):
        extract_research_figure_data(summary)


def test_reference_provider_diagnostic_rejects_inconsistent_accounting() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_diagnostic"][
        "provider_scalar_reduction_fraction"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="inconsistent scalar accounting",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assessment_fidelity_and_structure_gates_passed", 10),
        ("assessment_collision_panel_gate_passed", True),
        ("assessment_claim_consumed", False),
    ],
)
def test_reference_provider_diagnostic_preserves_sealed_decision(
    field: str,
    value: object,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_diagnostic"][field] = value

    with pytest.raises(
        ValueError,
        match="fidelity-pass and collision-control-fail",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    "field",
    [
        "natural_prompt_transfer_tested",
        "nll_measured",
        "model_parameter_compression_claim",
        "latency_measured",
    ],
)
def test_reference_provider_diagnostic_preserves_claim_boundaries(
    field: str,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_diagnostic"][field] = True

    with pytest.raises(
        ValueError,
        match="must preserve natural-prompt, NLL",
    ):
        extract_research_figure_data(summary)


def test_collision_attenuation_rejects_inconsistent_accounting() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_collision_attenuation_diagnostic"][
        "families"
    ][0]["pair_count"] = 9

    with pytest.raises(
        ValueError,
        match="inconsistent aggregate accounting",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_predictions_entered_collision_metric", True),
        ("assessment_panel_previously_opened", False),
        ("new_sealed_panel_opened", True),
        ("assessment_score_recomputed", True),
        ("candidate_refit_performed", True),
        ("candidate_reselection_performed", True),
        ("candidate_tracking_failure_can_be_assigned", True),
        ("formal_v2_decision_changed", True),
        ("target_derived_vjp_may_become_compiler_input", True),
        ("fresh_v3_assessment_required", False),
    ],
)
def test_collision_attenuation_preserves_v2_and_v3_boundary(
    field: str,
    value: object,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_collision_attenuation_diagnostic"][
        field
    ] = value

    with pytest.raises(
        ValueError,
        match="preserve frozen v2 and require a fresh v3",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "numerically_valid_gate_witness_count",
            15,
            "authenticated retrospective v2 teacher-path result",
        ),
        (
            "minimum_failed_witness_retained_64_fisher_energy_fraction",
            0.5,
            "reported family and mechanism result",
        ),
    ],
)
def test_collision_attenuation_preserves_numerical_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_collision_attenuation_diagnostic"][
        field
    ] = value

    with pytest.raises(ValueError, match=message):
        extract_research_figure_data(summary)


def test_v3_assessment_rejects_inconsistent_family_accounting() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_v3_assessment"]["contrast_families"][0][
        "candidate_pass_count"
    ] = 1

    with pytest.raises(
        ValueError,
        match="preserve sealed contrast-family accounting",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordinary_fidelity_passed", False),
        ("formal_outcome", "candidate_pass"),
        ("provider_passed", True),
    ],
)
def test_v3_assessment_preserves_mixed_formal_result(
    field: str,
    value: object,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_v3_assessment"][field] = value

    with pytest.raises(
        ValueError,
        match="ordinary-fidelity pass and panel-inconclusive",
    ):
        extract_research_figure_data(summary)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_parameters_changed",
        "natural_prompt_transfer_tested",
        "nll_measured",
        "whole_model_replacement_tested",
        "model_parameter_compression_claim",
        "latency_measured",
    ],
)
def test_v3_assessment_preserves_claim_boundaries(field: str) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["reference_provider_v3_assessment"][field] = True

    with pytest.raises(
        ValueError,
        match="metric-isolation and downstream claim boundaries",
    ):
        extract_research_figure_data(summary)


def test_dark_mode_preserves_pill_and_callout_contrast() -> None:
    (
        expected_ladder,
        expected_diagnostic,
        expected_bilinear,
        expected_attenuation,
        expected_v3_assessment,
        _,
    ) = _render_expected()

    assert 'class="status"' in expected_ladder
    assert 'style="fill:#166534"' in expected_ladder
    assert ".verdict-good { fill: #6ee7b7; }" in expected_diagnostic
    assert ".verdict-bad { fill: #fca5a5; }" in expected_diagnostic
    assert 'class="metric-scale verdict-good"' in expected_diagnostic
    assert 'class="metric-scale verdict-bad"' in expected_diagnostic
    assert ".callout { fill: #3b2616; stroke: #c2410c; }" in (
        expected_diagnostic
    )
    assert 'class="callout"' in expected_diagnostic
    assert "PASSES FROZEN ASSESSMENT" in expected_bilinear
    assert ".verdict-good { fill: #6ee7b7; }" in expected_bilinear
    assert "FORMAL V2 FAILURE UNCHANGED" in expected_attenuation
    assert "LOCALIZATION, NOT A RESCUE" in expected_attenuation
    assert ".verdict-bad { fill: #fca5a5; }" in expected_attenuation
    assert "6.77%" in expected_v3_assessment
    assert "0.9977" in expected_v3_assessment
    assert "72.7–130.3%" in expected_v3_assessment
    assert "cosine -0.922" in expected_v3_assessment
    assert "gain -2.443" in expected_v3_assessment
    assert "5 hallucinated changes" in expected_v3_assessment
    assert "PROVIDER PASSED: FALSE" in expected_v3_assessment
    assert ".verdict-bad { fill: #fca5a5; }" in expected_v3_assessment
