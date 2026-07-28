from __future__ import annotations

from collections import Counter

import pytest

from fisher_graph.gemma3_l3_l4_contrast_provider_development_protocol import (
    CALIBRATION_AMPLITUDE_GRID,
    CALIBRATION_EXACT_HALF_PAIRS,
    CONSUMED_C1_PILOT_PANEL_SHA256,
    CONSUMED_C1_PROTOCOL_SHA256,
    DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256,
    DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256,
    DEFAULT_DEVELOPMENT_PROTOCOL_SHA256,
    DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256,
    DEVELOPMENT_CANDIDATE_IDS,
    DEVELOPMENT_RANK_LADDER,
    CalibrationPilotMetric,
    DevelopmentCalibrationBinding,
    FrozenDevelopmentCandidateSet,
    default_contrast_provider_development_protocol,
    freeze_development_candidates,
    select_global_calibration_amplitude,
)
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    default_v3_assessment_protocol,
)
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    default_synthetic_reference_protocol,
)


_BANDS = (
    "band_00_07",
    "band_08_15",
    "band_16_31",
    "band_32_63",
)


def _pilot_metrics(
    *,
    eligible_from: float = 4.0,
    cosine: float = 0.99,
) -> tuple[CalibrationPilotMetric, ...]:
    exact_full_steps = {
        full for full, _half in CALIBRATION_EXACT_HALF_PAIRS
    }
    result = []
    for band in _BANDS:
        for amplitude in CALIBRATION_AMPLITUDE_GRID:
            label = str(amplitude).replace(".", "p")
            result.append(
                CalibrationPilotMetric(
                    metric_id=(
                        "development_c2.pilot.metric."
                        f"{band}.h_{label}"
                    ),
                    rank_band=band,  # type: ignore[arg-type]
                    amplitude=amplitude,
                    teacher_relative_effect_lower=(
                        0.03 if amplitude >= eligible_from else 0.01
                    ),
                    teacher_relative_effect_upper=0.10,
                    half_full_fd_cosine=(
                        cosine if amplitude in exact_full_steps else None
                    ),
                    half_full_fd_gain=(
                        1.0 if amplitude in exact_full_steps else None
                    ),
                )
            )
    return tuple(result)


def test_default_protocol_has_literal_hashes_and_40_80_80_geometry() -> None:
    protocol = default_contrast_provider_development_protocol()

    assert protocol.protocol_sha256 == DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
    assert (
        protocol.protocol_sha256
        == "033020dc9a0da819bd5753eb10090bff1bd9b4fcf61f33cd7186b1c1e3cb5254"
    )
    assert protocol.panel_sha256("pilot") == (
        DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256
    )
    assert protocol.panel_sha256("fit") == (
        DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256
    )
    assert protocol.panel_sha256("selection") == (
        DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256
    )
    assert DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256 == (
        "2b4d4efc8dbbdbc1c6afdb1a3134068f31b39ebc65e8a226fbbb9c92fe07b5e3"
    )
    assert DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256 == (
        "e6244896f2f6823f61d5ac57fad0756be320961e0b2272612579f9ba2f5b3e75"
    )
    assert DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256 == (
        "4e86a5acf6fcaa4a1e03b61e387938fb442f11e2d82aee73b56d1ee0189013f6"
    )
    assert tuple(
        len(protocol.probes_for_role(role))
        for role in ("pilot", "fit", "selection")
    ) == (40, 80, 80)
    assert tuple(
        len(protocol.groups_for_role(role))
        for role in ("pilot", "fit", "selection")
    ) == (20, 24, 24)
    assert protocol.rank_ladder == DEVELOPMENT_RANK_LADDER == (8, 16, 32)
    assert protocol.candidate_ids == DEVELOPMENT_CANDIDATE_IDS
    assert protocol.calibration_rule.amplitude_grid == (
        CALIBRATION_AMPLITUDE_GRID
    )
    assert protocol.calibration_rule.exact_half_pairs == (
        CALIBRATION_EXACT_HALF_PAIRS
    )
    assert protocol.calibration_rule.artifact_sha256 == (
        "7bb2e29e0baf3137ab1321811ecc4ed58f2d6068f1c94256ab26eda46eb14d10"
    )
    assert protocol.state_dict()["schema"] == (
        "fisher_graph.gemma3_l3_l4_contrast_provider_development_c2.v1"
    )
    predecessor = protocol.state_dict()["predecessor_pilot"]
    assert predecessor == {
        "protocol_sha256": CONSUMED_C1_PROTOCOL_SHA256,
        "pilot_panel_sha256": CONSUMED_C1_PILOT_PANEL_SHA256,
        "outcome": "failed_closed_no_eligible_global_amplitude",
        "maximum_tested_amplitude": 2.0,
        "teacher_relative_effect_at_maximum_by_band": {
            "band_00_07": 0.008251,
            "band_08_15": 0.003732,
            "band_16_31": 0.011773,
            "band_32_63": 0.003672,
        },
        "fd_cosine_summary": "approximately_one",
        "fd_gain_range_summary": [0.998, 0.999],
        "unchanged_minimum_effect_lower": 0.02,
        "failure_reason": "no_amplitude_met_unchanged_effect_floor",
        "fit_opened": False,
        "selection_opened": False,
        "c2_change_scope": (
            "fresh_pilot_identities_and_amplitude_grid_not_gate_tuning"
        ),
    }


def test_fit_and_selection_have_expected_families_pairs_and_bands() -> None:
    protocol = default_contrast_provider_development_protocol()
    expected = {
        "multitone": 8,
        "block_sparse": 8,
        "radial_sensitivity": 24,
        "signed_sensitivity": 16,
        "null_invariance": 24,
    }
    for role in ("fit", "selection"):
        probes = protocol.probes_for_role(role)  # type: ignore[arg-type]
        assert Counter(value.family for value in probes) == expected
        groups = protocol.groups_for_role(role)  # type: ignore[arg-type]
        assert Counter(value.family for value in groups) == {
            "radial_sensitivity": 8,
            "signed_sensitivity": 8,
            "null_invariance": 8,
        }
        assert sum(
            len(value.canonical_variant_pairs) for value in groups
        ) == 48
        for family in (
            "radial_sensitivity",
            "signed_sensitivity",
            "null_invariance",
        ):
            assert Counter(
                value.rank_band
                for value in groups
                if value.family == family
            ) == {band: 2 for band in _BANDS}

    pilot = protocol.probes_for_role("pilot")
    assert Counter(value.family for value in pilot) == {
        "calibration_signed": 40
    }
    assert Counter(
        (value.rank_band, value.modal_amplitude) for value in pilot
    ) == {
        (band, amplitude): 2
        for band in _BANDS
        for amplitude in CALIBRATION_AMPLITUDE_GRID
    }


def test_development_identities_are_disjoint_from_v2_and_v3() -> None:
    development = default_contrast_provider_development_protocol()
    v2 = default_synthetic_reference_protocol()
    v3 = default_v3_assessment_protocol()

    development_ids = {value.probe_id for value in development.probes}
    development_hashes = {
        value.artifact_sha256 for value in development.probes
    }
    development_seeds = {value.direction_seed for value in development.probes}
    assert not development_ids.intersection(
        value.probe_id for value in v2.probes
    )
    assert not development_ids.intersection(
        value.probe_id for value in v3.probes
    )
    assert not development_hashes.intersection(
        value.artifact_sha256 for value in v2.probes
    )
    assert not development_hashes.intersection(
        value.artifact_sha256 for value in v3.probes
    )
    assert not development_seeds.intersection(
        value.direction_seed for value in v2.probes
    )
    assert not development_seeds.intersection(
        value.direction_seed for value in v3.probes
    )
    consumed_lengths = {
        32,
        40,
        48,
        72,
        80,
        88,
        112,
        128,
        144,
        152,
        176,
        208,
        232,
        240,
        256,
    }
    assert not consumed_lengths.intersection(
        value.sequence_length for value in development.probes
    )


def test_c2_pilot_is_disjoint_from_consumed_c1_pilot() -> None:
    protocol = default_contrast_provider_development_protocol()
    pilot = protocol.probes_for_role("pilot")
    c1_geometry = {
        (4, 64, 9, 5),
        (10, 104, 31, 7),
        (22, 160, 73, 9),
        (47, 224, 137, 11),
    }
    c1_seeds = {
        4058646542285142682,
        7995415042857931943,
        10830807125392815439,
        13801626374411443275,
    }
    c2_geometry = {
        (
            value.axis_mode,
            value.sequence_length,
            value.source_offset,
            value.active_block_length,
        )
        for value in pilot
    }

    assert protocol.panel_sha256("pilot") != CONSUMED_C1_PILOT_PANEL_SHA256
    assert all(value.probe_id.startswith("development_c2.") for value in pilot)
    assert c2_geometry.isdisjoint(c1_geometry)
    assert {
        value.direction_seed for value in pilot
    }.isdisjoint(c1_seeds)
    assert c2_geometry == {
        (2, 68, 13, 6),
        (12, 108, 37, 8),
        (27, 164, 81, 10),
        (52, 228, 149, 12),
    }


def test_calibration_selects_smallest_global_eligible_amplitude() -> None:
    protocol = default_contrast_provider_development_protocol()
    binding = select_global_calibration_amplitude(
        protocol,
        _pilot_metrics(eligible_from=4.0),
    )

    assert isinstance(binding, DevelopmentCalibrationBinding)
    assert binding.selected_amplitude == 4.0
    assert binding.protocol_sha256 == protocol.protocol_sha256
    assert binding.pilot_panel_sha256 == protocol.panel_sha256("pilot")
    assert binding.artifact_sha256 == (
        "eab0f851338ccd3e2533531c2b46fd95d3cdba8184f2b1b6d1515ad7a278825b"
    )
    assert protocol.calibrated_panel_sha256("fit", binding) == (
        "586969180b05df7390cc24b14dea3ada9acad8aa4c701cbd8579f4c7730d11da"
    )
    assert protocol.calibrated_panel_sha256("selection", binding) == (
        "3c3e56eae65f4fb09645e91e8b0f297733172e15ef2642f58b7e720353a3a6fa"
    )


def test_amplitudes_without_exact_halves_cannot_be_selected() -> None:
    protocol = default_contrast_provider_development_protocol()
    binding = select_global_calibration_amplitude(
        protocol,
        _pilot_metrics(eligible_from=2.0),
    )
    assert binding.selected_amplitude == 4.0
    with pytest.raises(ValueError, match="must not claim FD metrics"):
        CalibrationPilotMetric(
            metric_id="development_c2.pilot.metric.invalid.h_2p0",
            rank_band="band_00_07",
            amplitude=2.0,
            teacher_relative_effect_lower=0.03,
            teacher_relative_effect_upper=0.10,
            half_full_fd_cosine=0.99,
            half_full_fd_gain=1.0,
        )


def test_calibration_fails_closed_when_stability_or_effect_rule_fails() -> None:
    protocol = default_contrast_provider_development_protocol()
    with pytest.raises(ValueError, match="no globally eligible"):
        select_global_calibration_amplitude(
            protocol,
            _pilot_metrics(eligible_from=13.0),
        )
    with pytest.raises(ValueError, match="no globally eligible"):
        select_global_calibration_amplitude(
            protocol,
            _pilot_metrics(eligible_from=4.0, cosine=0.97),
        )


def test_candidate_freeze_binds_exact_rank_ladder_fit_and_calibration() -> None:
    protocol = default_contrast_provider_development_protocol()
    calibration = select_global_calibration_amplitude(
        protocol,
        _pilot_metrics(),
    )
    frozen = freeze_development_candidates(
        protocol,
        calibration,
        ("1" * 64, "2" * 64, "3" * 64),
    )

    assert isinstance(frozen, FrozenDevelopmentCandidateSet)
    assert frozen.rank_ladder == (8, 16, 32)
    assert frozen.candidate_ids == DEVELOPMENT_CANDIDATE_IDS
    assert frozen.calibration_sha256 == calibration.artifact_sha256
    assert frozen.calibrated_fit_panel_sha256 == (
        protocol.calibrated_panel_sha256("fit", calibration)
    )
    assert frozen.artifact_sha256 == (
        "054529388698f7c2bdcfb443ddc26a6d78e2f7b42fb46271d753e69cd3326a6a"
    )
    with pytest.raises(ValueError, match="three unique"):
        freeze_development_candidates(
            protocol,
            calibration,
            ("1" * 64, "1" * 64, "3" * 64),
        )
