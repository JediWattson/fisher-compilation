from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_contrast_provider_development as runner
from fisher_graph.gemma3_l3_l4_contrast_provider_development_protocol import (
    CALIBRATION_AMPLITUDE_GRID,
    CALIBRATION_EXACT_HALF_PAIRS,
    default_contrast_provider_development_protocol,
    select_global_calibration_amplitude,
)


def _assert_tensor_free(value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("report metadata must not contain tensors")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_tensor_free(nested)


def test_parser_exposes_only_the_frozen_development_controls() -> None:
    parser = runner.build_parser()
    describe = parser.parse_args(["describe"])
    compile_args = parser.parse_args(["compile"])

    assert describe.command == "describe"
    assert compile_args.command == "compile"
    assert compile_args.basis_package == runner.DEFAULT_BASIS_PACKAGE
    assert compile_args.output == runner.DEFAULT_OUTPUT
    assert compile_args.device == "cpu"
    assert compile_args.dtype == "float32"

    for forbidden in ("--rank", "--steps", "--seed", "--model-id", "--force"):
        with pytest.raises(SystemExit):
            parser.parse_args(["compile", forbidden, "1"])


def test_describe_is_tensor_free_and_does_not_open_live_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe opened a model, materializer, or target")

    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)
    monkeypatch.setattr(runner, "materialize_development_role", forbidden)

    report = runner.describe_contrast_provider_development()

    assert report["schema"].endswith(".description")
    assert report["rank_ladder"] == (8, 16, 32)
    assert report["role_probe_counts"] == {
        "pilot": 40,
        "fit": 80,
        "selection": 80,
    }
    assert report["protocol_trust_anchor"] == report["protocol_sha256"]
    assert report["rank_semantics"] == (
        "all_64_modes_to_latent_r_to_all_64_modes"
    )
    for field in (
        "model_loaded",
        "pilot_materialized",
        "fit_materialized",
        "selection_materialized",
        "teacher_target_opened",
        "v2_targets_loaded",
        "v3_targets_loaded",
    ):
        assert report[field] is False
    _assert_tensor_free(report)


def test_calibration_metrics_preserve_exact_half_math_and_select_first_rung(
) -> None:
    protocol = default_contrast_provider_development_protocol()
    measured = []
    for probe in protocol.probes_for_role("pilot"):
        target = torch.zeros(
            1,
            probe.sequence_length,
            64,
            dtype=torch.float64,
        )
        target[..., 0] = 100.0
        target[..., 1] = 0.5 * probe.axis_sign * probe.modal_amplitude
        measured.append(
            SimpleNamespace(
                probe=probe,
                target_replays=(target, target.clone()),
                valid_mask=torch.ones(
                    1,
                    probe.sequence_length,
                    dtype=torch.bool,
                ),
            )
        )

    metrics = runner._calibration_metrics(
        protocol=protocol,
        measured=tuple(measured),
        metric_weight=torch.ones(64, dtype=torch.float64),
    )

    assert len(metrics) == 20
    exact_full_steps = {
        full_step for full_step, _half_step in CALIBRATION_EXACT_HALF_PAIRS
    }
    for metric in metrics:
        if metric.amplitude in exact_full_steps:
            assert metric.half_full_fd_cosine == pytest.approx(1.0)
            assert metric.half_full_fd_gain == pytest.approx(1.0)
        else:
            assert metric.half_full_fd_cosine is None
            assert metric.half_full_fd_gain is None
        expected_effect = metric.amplitude / (
            100.0**2 + (0.5 * metric.amplitude) ** 2
        ) ** 0.5
        assert metric.teacher_relative_effect_lower == pytest.approx(
            expected_effect,
            rel=1e-12,
            abs=1e-12,
        )
        assert metric.teacher_relative_effect_upper == pytest.approx(
            expected_effect,
            rel=1e-12,
            abs=1e-12,
        )

    binding = select_global_calibration_amplitude(protocol, metrics)
    assert binding.selected_amplitude == min(exact_full_steps)
    assert binding.selected_amplitude in CALIBRATION_AMPLITUDE_GRID


def test_combined_failure_reasons_are_complete_sorted_and_unique() -> None:
    ordinary = SimpleNamespace(
        passed=False,
        gate_flags=SimpleNamespace(
            state_dict=lambda: {
                "all_passed": False,
                "reference_cosine": False,
                "repeat_relative_error": False,
                "causality_violation": True,
            }
        ),
    )
    contrast = SimpleNamespace(
        overall_status="candidate_fail",
        reason_codes=("direction_error", "direction_error", "delta_error"),
    )

    reasons = runner._candidate_failure_reasons(
        ordinary_score=ordinary,
        contrast_result=contrast,
        coverage={"all_families_cover_all_four_rank_bands": False},
    )

    assert reasons == (
        "contrast:candidate_fail",
        "contrast:delta_error",
        "contrast:direction_error",
        "contrast:teacher_coverage_missing_rank_band",
        "ordinary:reference_cosine",
        "ordinary:repeat_relative_error",
    )
    assert runner._candidate_failure_reasons(
        ordinary_score=SimpleNamespace(passed=True),
        contrast_result=SimpleNamespace(
            overall_status="pass",
            reason_codes=(),
        ),
        coverage={"all_families_cover_all_four_rank_bands": True},
    ) == ()


def test_output_path_is_resolved_and_confined_inside_the_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.setattr(runner, "find_git_worktree", lambda _path: worktree)

    allowed = worktree / ".local-runs" / "result.pt"
    assert runner._validate_output_path(allowed) == allowed.resolve()

    with pytest.raises(ValueError, match="ignored local-runs"):
        runner._validate_output_path(worktree / "result.pt")
    with pytest.raises(ValueError, match=r"\.pt suffix"):
        runner._validate_output_path(worktree / ".local-runs" / "result.json")

    allowed.parent.mkdir()
    allowed.with_suffix(".json").touch()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._validate_output_path(allowed)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(runner, "find_git_worktree", lambda _path: None)
    shorthand = Path("~/outside.pt")
    assert runner._validate_output_path(shorthand) == (
        home / "outside.pt"
    ).resolve()


def test_packing_and_execution_diagnostics_are_tensor_free() -> None:
    rank = 8
    encoder = torch.linspace(
        0.1,
        1.0,
        steps=64 * rank,
        dtype=torch.float64,
    ).reshape(64, rank)
    decoder = torch.linspace(
        1.0,
        0.1,
        steps=rank * 64,
        dtype=torch.float64,
    ).reshape(rank, 64)
    plan = SimpleNamespace(
        encoder_weight=encoder,
        decoder_weight=decoder,
        latent_rank=rank,
        active_encoder_source_modes=64,
        active_decoder_target_modes=64,
    )

    packing = runner._mode_packing_diagnostics(plan)

    assert packing["latent_rank"] == rank
    assert packing["prefix_deletion_used"] is False
    assert packing["nonadjacent_modal_packing_available"] is True
    assert sum(
        packing["source_energy_fraction_by_rank_band"].values()
    ) == pytest.approx(1.0)
    assert sum(
        packing["target_energy_fraction_by_rank_band"].values()
    ) == pytest.approx(1.0)
    _assert_tensor_free(packing)

    class Runtime:
        def execution_accounting(
            self,
            *,
            valid_mask: torch.Tensor,
            logical_positions: torch.Tensor,
        ) -> object:
            assert valid_mask.shape == logical_positions.shape
            rows = int(valid_mask.sum())
            core = 11 * rows
            encoder_macs = 13 * rows
            decoder_macs = 17 * rows
            destandardization_macs = 2 * rows
            return SimpleNamespace(
                valid_rows=rows,
                core=SimpleNamespace(total_mac_count=core),
                encoder_mac_count=encoder_macs,
                decoder_mac_count=decoder_macs,
                target_destandardization_mac_count=destandardization_macs,
                total_mac_count=(
                    core
                    + encoder_macs
                    + decoder_macs
                    + destandardization_macs
                ),
            )

    accounting_plan = SimpleNamespace(
        prepare=lambda **_kwargs: Runtime(),
    )
    batches = (
        SimpleNamespace(
            batch=SimpleNamespace(
                valid_mask=torch.tensor([[True, True, False]]),
                logical_positions=torch.tensor([[0, 1, -1]]),
            )
        ),
        SimpleNamespace(
            batch=SimpleNamespace(
                valid_mask=torch.tensor([[True, False]]),
                logical_positions=torch.tensor([[4, -1]]),
            )
        ),
    )

    execution = runner._execution_accounting(accounting_plan, batches)

    assert execution["selection_panel_valid_rows"] == 3
    assert execution["macs_per_valid_row_over_selection_panel"] == 43
    assert execution["canonical_total_mac_count"] == 43 * 128
    _assert_tensor_free(execution)


def test_exact_hidden_chart_uses_nonlinear_midpoint_and_pushforward() -> None:
    left = torch.tensor([[[1.0]]], dtype=torch.float64)
    right = torch.tensor([[[3.0]]], dtype=torch.float64)

    def nonlinear_chart(
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            hidden.square(),
            hidden.pow(3),
            torch.exp(hidden[..., 0]),
        )

    primals, tangents = runner._exact_hidden_chart_midpoint_pushforward(
        nonlinear_chart,
        left_hidden=left,
        right_hidden=right,
    )
    left_chart = nonlinear_chart(left)
    right_chart = nonlinear_chart(right)
    endpoint_midpoints = tuple(
        0.5 * (left_value + right_value)
        for left_value, right_value in zip(
            left_chart,
            right_chart,
            strict=True,
        )
    )
    endpoint_chords = tuple(
        right_value - left_value
        for left_value, right_value in zip(
            left_chart,
            right_chart,
            strict=True,
        )
    )

    assert primals[0].item() == pytest.approx(4.0)
    assert endpoint_midpoints[0].item() == pytest.approx(5.0)
    assert primals[1].item() == pytest.approx(8.0)
    assert endpoint_midpoints[1].item() == pytest.approx(14.0)
    assert tangents[1].item() == pytest.approx(24.0)
    assert endpoint_chords[1].item() == pytest.approx(26.0)
    assert tangents[2].item() == pytest.approx(2.0 * torch.exp(
        torch.tensor(2.0, dtype=torch.float64)
    ).item())
    assert tangents[2].item() != pytest.approx(endpoint_chords[2].item())


def test_training_pairs_carry_exact_provider_chart_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    length = 2
    left = SimpleNamespace(
        probe=SimpleNamespace(probe_id="left", sequence_length=length),
        hidden_states=torch.zeros(1, length, 3, dtype=torch.float64),
        modal_coordinates=torch.zeros(1, length, 64, dtype=torch.float64),
        null_coordinates=torch.zeros(1, length, 1, dtype=torch.float64),
        row_rms=torch.ones(1, length, dtype=torch.float64),
    )
    right = SimpleNamespace(
        probe=SimpleNamespace(probe_id="right", sequence_length=length),
        hidden_states=torch.ones(1, length, 3, dtype=torch.float64),
        modal_coordinates=torch.ones(1, length, 64, dtype=torch.float64),
        null_coordinates=torch.ones(1, length, 1, dtype=torch.float64),
        row_rms=2.0 * torch.ones(1, length, dtype=torch.float64),
    )
    exact = runner._ProviderChartMidpointJVP(
        modal_primal=2.0 * torch.ones(length, 64, dtype=torch.float64),
        null_primal=3.0 * torch.ones(length, 1, dtype=torch.float64),
        row_rms_primal=4.0 * torch.ones(length, dtype=torch.float64),
        modal_tangent=5.0 * torch.ones(length, 64, dtype=torch.float64),
        null_tangent=6.0 * torch.ones(length, 1, dtype=torch.float64),
        row_rms_tangent=7.0 * torch.ones(length, dtype=torch.float64),
    )
    teacher = 8.0 * torch.ones(length, 64, dtype=torch.float64)
    monkeypatch.setattr(
        runner,
        "_provider_chart_midpoint_jvp",
        lambda **_kwargs: exact,
    )
    monkeypatch.setattr(
        runner,
        "_teacher_midpoint_jvp",
        lambda **_kwargs: teacher,
    )
    group = SimpleNamespace(
        group_id="fit.synthetic",
        family="nonlinear_synthetic",
        intent="sensitivity",
        rank_band="32-63",
        canonical_variant_pairs=(("left", "right"),),
    )
    protocol = SimpleNamespace(
        groups_for_role=lambda role: (group,) if role == "fit" else (),
    )

    pairs, diagnostics = runner._training_contrast_pairs(
        protocol=protocol,
        measured=(left, right),
        basis=object(),
        adapter=object(),
        pre_ff3=object(),
        post_ff3=object(),
        epsilon=1e-6,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert torch.equal(pair.teacher_midpoint_jvp, teacher)
    assert torch.equal(pair.provider_chart_modal_primal, exact.modal_primal)
    assert torch.equal(pair.provider_chart_null_primal, exact.null_primal)
    assert torch.equal(
        pair.provider_chart_row_rms_primal,
        exact.row_rms_primal,
    )
    assert torch.equal(pair.provider_chart_modal_tangent, exact.modal_tangent)
    assert torch.equal(pair.provider_chart_null_tangent, exact.null_tangent)
    assert torch.equal(
        pair.provider_chart_row_rms_tangent,
        exact.row_rms_tangent,
    )
    assert len(diagnostics) == 1
    assert diagnostics[0]["endpoint_arithmetic_used_for_fit"] is False
    _assert_tensor_free(diagnostics)
