from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

import fisher_graph.gemma3_l3_l4_attenuation_localization_experiment as runner
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    default_synthetic_reference_protocol,
)


def test_defaults_bind_the_frozen_failed_v2_artifacts() -> None:
    assert runner.DEFAULT_CANDIDATE.name.endswith(
        "l3-l4-reference-provider-dev-v2.pt"
    )
    assert runner.DEFAULT_ASSESSMENT.name.endswith(
        "l3-l4-reference-provider-assessment-dev-v2.pt"
    )
    assert len(runner.DEFAULT_CANDIDATE_FILE_SHA256) == 64
    assert len(runner.DEFAULT_CANDIDATE_REPORT_SHA256) == 64
    assert len(runner.DEFAULT_ASSESSMENT_FILE_SHA256) == 64
    assert len(runner.DEFAULT_ASSESSMENT_REPORT_SHA256) == 64
    assert runner._COLLISION_THRESHOLD == 0.01
    assert runner._CONTRAST_THRESHOLDS.state_dict() == {
        "numeric_floor_epsilon_multiplier": 4.0,
        "resolved_noise_multiplier": 8.0,
        "maximum_linearization_relative_error": 0.25,
        "maximum_adjoint_relative_error": 0.0001,
        "maximum_causal_leakage_fraction": 1e-10,
        "localization_contraction_ratio": 0.25,
        "localization_dominance_ratio": 0.5,
    }


def test_collision_panel_geometry_has_40_endpoints_and_32_pairs() -> None:
    probes = runner._collision_probes(
        default_synthetic_reference_protocol()
    )
    assert len(probes) == 40
    assert Counter(probe.family for probe in probes) == {
        "axis": 16,
        "radial_collision": 12,
        "null_collision": 12,
    }
    grouped: dict[str, list[object]] = defaultdict(list)
    for probe in probes:
        assert probe.collision_group is not None
        grouped[probe.collision_group].append(probe)
    assert len(grouped) == 16
    assert Counter(len(group) for group in grouped.values()) == {2: 8, 3: 8}
    assert sum(
        len(group) * (len(group) - 1) // 2
        for group in grouped.values()
    ) == 32


def test_distribution_is_deterministic() -> None:
    assert runner._distribution((4.0, 1.0, 3.0, 2.0)) == {
        "minimum": 1.0,
        "median": 2.5,
        "p90": pytest.approx(3.7),
        "maximum": 4.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        runner._distribution(())


def test_output_path_must_be_local_pt_and_non_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.setattr(runner, "find_git_worktree", lambda _path: worktree)

    allowed = worktree / ".local-runs" / "trace.pt"
    assert runner._validate_output_path(allowed) == allowed
    with pytest.raises(ValueError, match="must remain under"):
        runner._validate_output_path(worktree / "trace.pt")
    with pytest.raises(ValueError, match="must use .pt"):
        runner._validate_output_path(worktree / ".local-runs" / "trace.json")

    allowed.parent.mkdir()
    allowed.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._validate_output_path(allowed)


def test_parser_keeps_the_diagnostic_on_cpu_by_default() -> None:
    arguments = runner.build_parser().parse_args([])
    assert arguments.candidate == runner.DEFAULT_CANDIDATE
    assert arguments.assessment == runner.DEFAULT_ASSESSMENT
    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.device == "cpu"
    assert arguments.dtype == "float32"


def test_unresolved_zero_contrast_does_not_invent_a_mechanism() -> None:
    checkpoints = tuple(
        {
            "checkpoint_name": name,
            "resolved": False,
            "jvp_l2": 0.0,
            "midpoint_jvp_relative_response": 0.0,
            "numeric_floor_l2": 1e-6,
            "left_l2": 0.0,
            "right_l2": 0.0,
            "midpoint_linearization_relative_error": 0.0,
            "symmetric_relative_separation": 0.0,
        }
        for name in runner._CHECKPOINT_NAMES
    )
    pair = runner._Pair(
        pair_id="zero",
        family="null_collision",
        collision_group="zero",
        left_variant="left",
        right_variant="right",
        left_probe_id="left",
        right_probe_id="right",
        sequence_length=2,
        source_offset=1,
        target_relative_difference=0.0,
        gate_witness=True,
    )
    summary = runner._mechanism_summary(
        pair=pair,
        checkpoints=checkpoints,
        cancellation={
            "residual_attention_secant_cosine": None,
            "merge_cancellation_ratio": 0.0,
            "residual_additivity_relative_error": 0.0,
        },
        fisher_capture={
            "full_width_fisher_weighted_secant_l2": 0.0,
            "retained_64_fisher_weighted_secant_l2": 0.0,
            "retained_64_fisher_energy_fraction": 0.0,
        },
        adjoint_relative_error=0.0,
        causal_leakage_fraction=0.0,
    )
    assert summary["teacher_status"] == "numerically_unresolved"
    assert summary["mechanism_observations"] == ()
    assert not summary["diagnostic_numerically_valid"]
    assert (
        summary[
            "dominant_midpoint_jvp_relative_response_contraction_transition"
        ]
        is None
    )


def test_causal_gate_uses_target_path_not_an_unrelated_side_branch() -> None:
    checkpoints = [
        {
            "checkpoint_name": name,
            "jvp_energy_regions": {"pre_source_fraction": 0.0},
        }
        for name in runner._CHECKPOINT_NAMES
    ]
    by_name = {row["checkpoint_name"]: row for row in checkpoints}
    by_name["layer.3.mlp.normalized_input"]["jvp_energy_regions"][
        "pre_source_fraction"
    ] = 1.0
    assert runner._target_causal_leakage_fraction(checkpoints) == 0.0

    by_name["l4.fisher_weighted_target"]["jvp_energy_regions"][
        "pre_source_fraction"
    ] = 0.125
    assert runner._target_causal_leakage_fraction(checkpoints) == 0.125
