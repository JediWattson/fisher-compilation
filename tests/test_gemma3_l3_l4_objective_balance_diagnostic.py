from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_objective_balance_diagnostic as runner
from fisher_graph.gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    default_objective_balance_diagnostic_protocol,
    family_balance_copy_binding_sha256,
    family_balance_copy_id,
)
from fisher_graph.state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastTrainingMetrics,
    IndexedReferenceBatch,
    ReferenceProviderContrastPair,
)
from fisher_graph.state_conditioned_reference_provider import (
    SyntheticReferenceBatch,
)
from fisher_graph.state_conditioned_reference_selection import (
    FullWidthCandidatePrediction,
    FullWidthReferenceCandidate,
    FullWidthReferenceProbe,
    FullWidthStructuralMetrics,
)


def _assert_tensor_free(value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("report metadata contains a Tensor")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_tensor_free(nested)


def _evaluation(
    recipe_id: str,
    seed_role: str,
    seed: int,
    *,
    passed: bool,
) -> runner._RecipeEvaluation:
    return runner._RecipeEvaluation(
        recipe_id=recipe_id,
        seed_role=seed_role,
        seed=seed,
        combined_pass=passed,
        row={"candidate_id": f"{recipe_id}.{seed_role}"},
        plan=SimpleNamespace(artifact_sha256="a" * 64),
    )


def test_parser_exposes_only_describe_and_frozen_run_controls() -> None:
    parser = runner.build_parser()
    describe = parser.parse_args(["describe"])
    run = parser.parse_args(["run"])

    assert describe.command == "describe"
    assert run.command == "run"
    assert run.basis_package == runner.DEFAULT_BASIS_PACKAGE
    assert run.output == runner.DEFAULT_OUTPUT
    assert run.device == "cpu"
    assert run.dtype == "float32"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--device", "mps"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--dtype", "bfloat16"])
    for forbidden in ("--recipe", "--steps", "--seed", "--rank", "--force"):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", forbidden, "1"])


def test_describe_is_tensor_free_and_opens_no_live_or_selection_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe opened live data")

    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)
    monkeypatch.setattr(runner, "_measure_c2_role", forbidden)

    report = runner.describe_objective_balance_diagnostic()

    assert report["protocol_sha256"] == report["protocol_trust_anchor"]
    assert report["recipe_ids"] == (
        "d0_raw_c2_control",
        "d1_unit_rms",
        "d2_unit_rms_family_balanced",
        "d3_unit_rms_family_balanced_direction",
    )
    assert report["allowed_c2_role_probe_counts"] == {
        "pilot": 40,
        "fit": 80,
    }
    assert "continue_after_failed_replication" in report["decision_rule"]
    for field in (
        "selection_role_allowed",
        "selection_materialization_allowed",
        "selection_measurement_allowed",
        "c2_artifact_loading_allowed",
        "model_loaded",
        "pilot_materialized",
        "fit_materialized",
        "teacher_target_opened",
        "selection_materialized",
        "selection_target_opened",
        "v2_targets_loaded",
        "v3_targets_loaded",
    ):
        assert report[field] is False
    _assert_tensor_free(report)


def test_run_rejects_nonfrozen_execution_before_live_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution mismatch reached live loading")

    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)

    with pytest.raises(ValueError, match="frozen to cpu/float32"):
        runner.run_objective_balance_diagnostic(device_name="mps")
    with pytest.raises(ValueError, match="frozen to cpu/float32"):
        runner.run_objective_balance_diagnostic(dtype="bfloat16")


def test_c2_role_firewall_rejects_selection_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    def measure(*, role: str, **_kwargs: object):
        delegated.append(role)
        return (), {"role": role}

    monkeypatch.setattr(runner.c2, "_measure_role", measure)

    assert runner._measure_c2_role(role="pilot") == (
        (),
        {"role": "pilot"},
    )
    assert runner._measure_c2_role(role="fit") == ((), {"role": "fit"})
    with pytest.raises(PermissionError, match="forbids C2 selection"):
        runner._measure_c2_role(role="selection")
    assert delegated == ["pilot", "fit"]


def test_authenticated_protocol_binds_executed_d0_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_objective_for_recipe",
        lambda _recipe: SimpleNamespace(artifact_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="C2 provenance drifted"):
        runner._authenticated_protocols()


def test_fit_only_ordinary_scorer_preserves_split_and_uses_real_gate_core(
) -> None:
    gauge_sha256 = "a" * 64
    target = torch.zeros(1, 4, 64, dtype=torch.float64)
    target[0, :, 0] = torch.tensor(
        [1.0, 2.0, 4.0, 8.0],
        dtype=torch.float64,
    )
    target[0, :, 1] = torch.tensor(
        [-2.0, 3.0, -1.0, 5.0],
        dtype=torch.float64,
    )
    probe = FullWidthReferenceProbe(
        probe_id="development_c2.fit.ordinary.00",
        split="fit",
        family="multitone",
        standardized_target=target,
        logical_positions=torch.arange(4).view(1, -1),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        standardized_gauge_sha256=gauge_sha256,
    )
    controls = runner.fit_full_width_reference_controls(
        fit_probes=(probe,),
        position_bin_count=1,
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id="fit-only",
        source_rank=64,
        target_rank=64,
        stored_scalar_count=1,
        predictions=(
            FullWidthCandidatePrediction(
                probe_id=probe.probe_id,
                retained_standardized_prediction=target,
                standardized_gauge_sha256=gauge_sha256,
            ),
        ),
        structural_metrics=FullWidthStructuralMetrics(
            prepared_vs_analytic_relative_error=0.0,
            causality_violation=0.0,
            padding_violation=0.0,
            repeat_relative_error=0.0,
            in_support_fraction=1.0,
        ),
        candidate_binding_sha256="b" * 64,
    )

    score = runner._score_fit_only_ordinary_candidate(
        controls=controls,
        fit_probes=(probe,),
        candidate=candidate,
        gates=runner.SyntheticReferenceGates(),
    )

    assert probe.split == "fit"
    assert score.passed
    assert score.fisher_weighted_relative_error == 0.0
    flags = score.gate_flags.state_dict()
    assert len(flags) == 12
    assert score.gate_flags.all_passed is True
    assert all(flags.values())

    changed_probe = FullWidthReferenceProbe(
        probe_id=probe.probe_id,
        split="fit",
        family=probe.family,
        standardized_target=target * 2.0,
        logical_positions=probe.logical_positions,
        valid_mask=probe.valid_mask,
        standardized_gauge_sha256=gauge_sha256,
    )
    with pytest.raises(ValueError, match="hashes differ"):
        runner._score_fit_only_ordinary_candidate(
            controls=controls,
            fit_probes=(changed_probe,),
            candidate=candidate,
            gates=runner.SyntheticReferenceGates(),
        )


def test_schedule_continues_after_failed_replication_and_stops_on_two_seed_pass(
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    calls: list[tuple[str, str]] = []

    def evaluate(recipe, seed_role: str, seed: int):
        calls.append((recipe.recipe_id, seed_role))
        passed = (
            recipe.recipe_id in {
                "d1_unit_rms",
                "d2_unit_rms_family_balanced",
            }
            if seed_role == "primary"
            else recipe.recipe_id == "d2_unit_rms_family_balanced"
        )
        return _evaluation(
            recipe.recipe_id,
            seed_role,
            seed,
            passed=passed,
        )

    results, selected, replication = runner._execute_recipe_schedule(
        protocol,
        evaluate=evaluate,
    )

    assert calls == [
        ("d0_raw_c2_control", "primary"),
        ("d1_unit_rms", "primary"),
        ("d1_unit_rms", "replication"),
        ("d2_unit_rms_family_balanced", "primary"),
        ("d2_unit_rms_family_balanced", "replication"),
    ]
    assert len(results) == 5
    assert selected is not None
    assert selected.recipe_id == "d2_unit_rms_family_balanced"
    assert replication is not None and replication.combined_pass


def test_schedule_exhausts_all_primary_treatments_when_none_pass() -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    calls: list[tuple[str, str]] = []

    def evaluate(recipe, seed_role: str, seed: int):
        calls.append((recipe.recipe_id, seed_role))
        return _evaluation(
            recipe.recipe_id,
            seed_role,
            seed,
            passed=False,
        )

    results, selected, replication = runner._execute_recipe_schedule(
        protocol,
        evaluate=evaluate,
    )

    assert [role for _, role in calls] == ["primary"] * 4
    assert [recipe for recipe, _ in calls] == [
        value.recipe_id for value in protocol.recipes
    ]
    assert len(results) == 4
    assert selected is None
    assert replication is None


def test_decision_distinguishes_failed_replication_from_no_primary_pass(
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    d0 = _evaluation(
        "d0_raw_c2_control",
        "primary",
        protocol.primary_seed,
        passed=False,
    )
    d1 = _evaluation(
        "d1_unit_rms",
        "primary",
        protocol.primary_seed,
        passed=True,
    )
    d1_replication = _evaluation(
        "d1_unit_rms",
        "replication",
        protocol.replication_seed,
        passed=False,
    )

    failed_replication = runner._diagnostic_decision(
        protocol,
        (d0, d1, d1_replication),
        two_seed_primary=None,
        two_seed_replication=None,
    )
    no_primary = runner._diagnostic_decision(
        protocol,
        (d0,),
        two_seed_primary=None,
        two_seed_replication=None,
    )

    assert failed_replication["outcome"] == (
        "primary_fit_passes_failed_replication_no_c3_authority"
    )
    assert failed_replication["first_primary_passing_recipe_id"] == (
        "d1_unit_rms"
    )
    assert failed_replication["authorized_fresh_c3_recipe_id"] is None
    assert no_primary["outcome"] == (
        "no_primary_treatment_passed_fit_gates"
    )


def _natural_pairs() -> tuple[ReferenceProviderContrastPair, ...]:
    result = []
    for family, count, role in (
        ("radial_sensitivity", 16, "expected_sensitivity"),
        ("signed_sensitivity", 8, "expected_sensitivity"),
        ("null_invariance", 24, "intended_null"),
    ):
        for index in range(count):
            result.append(
                ReferenceProviderContrastPair(
                    pair_id=f"fit.{family}.{index:02d}",
                    family=family,
                    role=role,
                    left_endpoint_id=f"{family}.left.{index:02d}",
                    right_endpoint_id=f"{family}.right.{index:02d}",
                    rank_stratum=f"band.{index % 4}",
                )
            )
    return tuple(result)


def test_family_balance_duplicates_only_signed_pairs_with_protocol_bindings(
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    recipe = protocol.recipe("d2_unit_rms_family_balanced")
    natural = _natural_pairs()

    balanced, report = runner._balance_training_pairs(
        natural,
        recipe=recipe,
    )

    assert len(balanced) == 56
    assert report["natural_family_counts"] == {
        "radial_sensitivity": 16,
        "signed_sensitivity": 8,
        "null_invariance": 24,
    }
    assert report["balanced_family_counts"] == {
        "radial_sensitivity": 16,
        "signed_sensitivity": 16,
        "null_invariance": 24,
    }
    originals = {
        value.pair_id: value
        for value in natural
        if value.family == "signed_sensitivity"
    }
    duplicate_bindings = report["duplicate_bindings"]
    assert len(duplicate_bindings) == 8
    for binding in duplicate_bindings:
        source = originals[binding["source_pair_id"]]
        assert binding["copy_binding_sha256"] == (
            family_balance_copy_binding_sha256(
                source.pair_id,
                source.artifact_sha256,
            )
        )
        assert binding["duplicate_pair_id"] == family_balance_copy_id(
            source.pair_id,
            source.artifact_sha256,
        )
    _assert_tensor_free(report)


def test_recipe_training_metric_drives_floor_diagnostic_gauge() -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    raw = torch.linspace(2.0, 65.0, 64, dtype=torch.float64)
    gauge = runner.UnitRmsFisherGauge.from_metric_weight(raw)

    control_metric = runner._training_metric_for_recipe(
        protocol.recipe("d0_raw_c2_control"),
        raw_metric_weight=raw,
        unit_rms_gauge=gauge,
    )
    treatment_metric = runner._training_metric_for_recipe(
        protocol.recipe("d1_unit_rms"),
        raw_metric_weight=raw,
        unit_rms_gauge=gauge,
    )

    assert torch.equal(control_metric, raw)
    assert torch.equal(treatment_metric, gauge.metric_weight)
    assert torch.sqrt(treatment_metric.square().mean()).item() == (
        pytest.approx(1.0, rel=1e-14, abs=1e-14)
    )
    assert not torch.equal(control_metric, treatment_metric)


def test_fit_data_binding_is_common_while_recipe_bindings_differ() -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    c2_protocol = (
        runner.default_contrast_provider_development_protocol()
    )
    basis = SimpleNamespace(
        basis_payload_sha256="a" * 64,
        source_model_sha256="b" * 64,
    )
    calibration = SimpleNamespace(artifact_sha256="c" * 64)
    raw_metric = torch.linspace(1.0, 2.0, 64, dtype=torch.float64)
    gauge = runner.UnitRmsFisherGauge.from_metric_weight(raw_metric)
    common_bindings = []
    candidate_bindings = []
    for recipe in protocol.recipes:
        common_bindings.append(
            runner._fit_data_binding_sha256(
                basis=basis,
                c2_protocol=c2_protocol,
                calibration=calibration,
                norm_sha256="d" * 64,
                canonical_metric_weight=raw_metric,
            )
        )
        training_metric = runner._training_metric_for_recipe(
            recipe,
            raw_metric_weight=raw_metric,
            unit_rms_gauge=gauge,
        )
        candidate_bindings.append(
            runner._provider_binding_sha256(
                diagnostic_protocol_sha256=protocol.protocol_sha256,
                recipe=recipe,
                c2_protocol_sha256=c2_protocol.protocol_sha256,
                calibration_sha256=calibration.artifact_sha256,
                basis_payload_sha256=basis.basis_payload_sha256,
                source_model_sha256=basis.source_model_sha256,
                norm_sha256="d" * 64,
                training_metric_weight=training_metric,
                target_center=torch.zeros(64, dtype=torch.float64),
                target_scale=torch.ones(64, dtype=torch.float64),
            )
        )

    assert len(set(common_bindings)) == 1
    assert common_bindings[0] == runner.c2._provider_binding_sha256(
        basis=basis,
        protocol=c2_protocol,
        calibration=calibration,
        objective=runner.c2._objective(),
        norm_sha256="d" * 64,
        metric_weight=raw_metric,
    )
    assert len(set(candidate_bindings)) == 4


def test_contribution_gate_uses_recipe_metric_signal_and_declared_shares(
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    recipe = protocol.recipe("d1_unit_rms")
    objective = runner._objective_for_recipe(recipe)
    metrics = ContrastTrainingMetrics(
        pointwise_mse=0.20,
        sensitivity_relative_delta_mse=0.10,
        sensitivity_direction_loss=0.10,
        midpoint_jvp_relative_mse=0.30,
        intended_null_absolute_mse=0.01,
        weighted_total=0.76,
        endpoint_count=80,
        sensitivity_pair_count=24,
        jvp_pair_count=24,
        intended_null_pair_count=24,
    )
    plan = SimpleNamespace(
        initial_metrics=metrics,
        objective=objective,
        fisher_metric_supplied=True,
        fisher_metric_weight=torch.ones(64, dtype=torch.float64),
    )

    result = runner._contribution_balance_gate(
        plan,
        recipe=recipe,
        gates=protocol.gates,
        training_teacher_energy=1.0,
        raw_teacher_energy=10.0,
        teacher_signal_diagnostics={
            "minimum_teacher_delta_mse": 1e-3,
            "minimum_teacher_jvp_mse": 2e-3,
        },
    )

    assert result["passed"] is True
    assert all(result["flags"].values())
    assert result["initial_contribution_audit"][
        "reported_total_matches"
    ] is True

    unsafe_signal = runner._contribution_balance_gate(
        plan,
        recipe=recipe,
        gates=protocol.gates,
        training_teacher_energy=1.0,
        raw_teacher_energy=10.0,
        teacher_signal_diagnostics={
            "minimum_teacher_delta_mse": 1e-14,
            "minimum_teacher_jvp_mse": 2e-3,
        },
    )
    assert unsafe_signal["passed"] is False
    assert unsafe_signal["flags"][
        "teacher_delta_above_relative_floor"
    ] is False


def test_output_path_and_atomic_report_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.setattr(runner, "find_git_worktree", lambda _path: worktree)
    allowed = worktree / ".local-runs" / "diagnostic.pt"
    assert runner._validate_output_path(allowed) == allowed.resolve()
    with pytest.raises(ValueError, match="ignored local-runs"):
        runner._validate_output_path(worktree / "diagnostic.pt")
    with pytest.raises(ValueError, match=r"\.pt suffix"):
        runner._validate_output_path(
            worktree / ".local-runs" / "diagnostic.json"
        )

    output = tmp_path / "published.pt"
    report = runner._publish_artifact(
        {"provider_weight": torch.ones(2)},
        {"schema": "safe-test", "contains_raw_teacher_targets": False},
        output=output,
    )
    report_path = output.with_suffix(".json")
    assert output.is_file()
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["report_sha256"] == (
        report["report_sha256"]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._publish_artifact({}, {}, output=output)
    with pytest.raises(ValueError, match="forbidden"):
        runner._publish_artifact(
            {"teacher_midpoint_jvp": torch.ones(1)},
            {},
            output=tmp_path / "unsafe.pt",
        )


def test_post_publish_authentication_failure_removes_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rejected.pt"

    def reject(_path: Path) -> object:
        raise ValueError("post-publish validation failed")

    monkeypatch.setattr(
        runner,
        "load_objective_balance_diagnostic_artifact",
        reject,
    )

    with pytest.raises(ValueError, match="post-publish validation failed"):
        runner._publish_and_authenticate_artifact(
            {"safe": torch.ones(1)},
            {"schema": "rejected-test"},
            output=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_published_artifact_loader_is_weights_only_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    metric_weight = torch.linspace(1.0, 2.0, 64, dtype=torch.float64)
    gauge = runner.UnitRmsFisherGauge.from_metric_weight(metric_weight)
    code_sha256s = {"runner": "a" * 64}
    code_bundle_sha256 = "b" * 64
    monkeypatch.setattr(
        runner,
        "_code_sha256s",
        lambda: code_sha256s,
    )
    monkeypatch.setattr(
        runner,
        "_code_bundle_sha256",
        lambda _values: code_bundle_sha256,
    )
    c2_protocol = (
        runner.default_contrast_provider_development_protocol()
    )
    calibration = runner.DevelopmentCalibrationBinding(
        protocol_sha256=c2_protocol.protocol_sha256,
        pilot_panel_sha256=c2_protocol.panel_sha256("pilot"),
        calibration_rule_sha256=(
            c2_protocol.calibration_rule.artifact_sha256
        ),
        selected_amplitude=8.0,
        pilot_metric_sha256s=tuple(
            f"{index + 1:064x}" for index in range(20)
        ),
    )
    gauge_sha256 = "e" * 64
    control_target = torch.zeros(1, 2, 64, dtype=torch.float64)
    control_target[0, :, 0] = torch.tensor([1.0, 2.0])
    control_probe = FullWidthReferenceProbe(
        probe_id="fit.control",
        split="fit",
        family="multitone",
        standardized_target=control_target,
        logical_positions=torch.arange(2).view(1, -1),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        standardized_gauge_sha256=gauge_sha256,
    )
    controls = runner.fit_full_width_reference_controls(
        fit_probes=(control_probe,),
        position_bin_count=1,
    )
    modal = torch.zeros(2, 2, 64, dtype=torch.float64)
    modal[0, :, 0] = -1.0
    modal[1, :, 0] = 1.0
    indexed = IndexedReferenceBatch(
        batch=SyntheticReferenceBatch(
            split="fit",
            modal_coordinates=modal,
            null_coordinates=torch.zeros(2, 2, 1, dtype=torch.float64),
            row_rms=torch.ones(2, 2, dtype=torch.float64),
            target_modes=0.1 * modal,
            logical_positions=torch.arange(2).repeat(2, 1),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
            synthetic_binding_sha256="9" * 64,
        ),
        endpoint_ids=("left", "right"),
    )
    fit_pair = ReferenceProviderContrastPair(
        pair_id="fit.pair",
        family="signed",
        role="expected_sensitivity",
        left_endpoint_id="left",
        right_endpoint_id="right",
        rank_stratum="test",
    )
    plan = runner.fit_contrast_aware_reference_provider(
        modal_center=torch.zeros(64, dtype=torch.float64),
        gain_log_center=0.0,
        gain_log_scale=1.0,
        residual_width=64,
        rms_epsilon=1e-6,
        target_center=torch.zeros(64, dtype=torch.float64),
        target_scale=torch.ones(64, dtype=torch.float64),
        fit_batches=(indexed,),
        contrast_pairs=(fit_pair,),
        executor_config=runner.c2._executor_config(16),
        objective=ContrastAwareObjective(
            pointwise_weight=1.0,
            sensitivity_relative_delta_weight=0.0,
            sensitivity_direction_weight=0.0,
            midpoint_jvp_weight=0.0,
            intended_null_weight=0.0,
        ),
        fisher_metric_weight=torch.ones(64, dtype=torch.float64),
        steps=1,
        learning_rate=1e-3,
        seed=1,
    )
    candidate_id = "d0_raw_c2_control.primary"
    candidate_row = {
        "candidate_id": candidate_id,
        "combined_pass": False,
    }
    plan_sha256 = plan.artifact_sha256
    result_sha256 = runner._json_sha256(
        candidate_row,
        domain=runner._ARTIFACT_DOMAIN,
    )
    manifest = {
        "schema": runner._SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "protocol_sha256": protocol.artifact_sha256,
        "requested_execution_device": "cpu",
        "requested_execution_dtype": "float32",
        "actual_execution_device": "cpu",
        "actual_execution_dtype": "float32",
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_calibration_sha256": calibration.artifact_sha256,
        "selected_calibration_amplitude": 8.0,
        "canonical_metric_weight_sha256": runner._tensor_sha256(
            metric_weight
        ),
        "unit_rms_gauge_sha256": gauge.artifact_sha256,
        "standardized_gauge_sha256": gauge_sha256,
        "controls_sha256": controls.artifact_sha256,
        "executed_candidate_ids": (candidate_id,),
        "candidate_plan_sha256s": {candidate_id: plan_sha256},
        "candidate_result_sha256s": {candidate_id: result_sha256},
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_artifact_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": code_bundle_sha256,
        "outcome": "test_only",
    }
    logical_sha256 = runner._json_sha256(
        manifest,
        domain=runner._ARTIFACT_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": logical_sha256,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "unit_rms_gauge_state": gauge.state_dict(),
        "canonical_metric_weight": metric_weight,
        "controls_state": controls.state_dict(),
        "plan_states": {candidate_id: plan.state_dict()},
        "candidate_results": {candidate_id: candidate_row},
    }
    report_payload = {
        **manifest,
        "artifact_sha256": logical_sha256,
        "protocol": protocol.state_dict(),
        "calibration": calibration.state_dict(),
        "pilot_metrics": (),
        "pilot_measurement": {},
        "fit_measurement": {},
        "fit_provider_chart_mismatch_diagnostics": (),
        "teacher_signal_diagnostics": {},
        "gauge": {"artifact_sha256": gauge.artifact_sha256},
        "candidate_results": (candidate_row,),
        "interpretation": {"fit_side_only": True},
        "safety": {"committable": False},
    }
    output = tmp_path / "diagnostic.pt"
    runner._publish_artifact(state, report_payload, output=output)

    original_torch_load = torch.load
    load_calls: list[object] = []

    def recording_load(*args: object, **kwargs: object):
        load_calls.append(kwargs.get("weights_only"))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded = runner.load_objective_balance_diagnostic_artifact(output)

    assert load_calls == [True]
    assert loaded.artifact_sha256 == logical_sha256
    assert loaded.tensor_file_sha256 == loaded.report["artifact"][
        "tensor_file_sha256"
    ]
    assert loaded.report_sha256 == loaded.report["report_sha256"]

    report_path = output.with_suffix(".json")
    tampered_report = json.loads(report_path.read_text())
    tampered_report["outcome"] = "tampered"
    report_path.write_text(json.dumps(tampered_report))
    with pytest.raises(ValueError, match="report SHA-256 mismatch"):
        runner.load_objective_balance_diagnostic_artifact(output)

    second_output = tmp_path / "diagnostic-logical-tamper.pt"
    runner._publish_artifact(state, report_payload, output=second_output)
    tampered_state = original_torch_load(
        second_output,
        map_location="cpu",
        weights_only=True,
    )
    tampered_state["manifest"]["outcome"] = "tampered"
    original_torch_save = torch.save
    original_torch_save(tampered_state, second_output)
    with pytest.raises(ValueError, match="logical artifact binding mismatch"):
        runner.load_objective_balance_diagnostic_artifact(second_output)


def test_run_orchestration_uses_only_c2_pilot_and_fit_and_same_fit_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    provenance = protocol.c2_provenance

    class C2Protocol:
        protocol_sha256 = provenance.protocol_sha256

        @staticmethod
        def panel_sha256(role: str) -> str:
            if role == "pilot":
                return provenance.pilot_panel_sha256
            if role == "fit":
                return provenance.fit_panel_sha256
            raise AssertionError("run requested C2 selection")

        @staticmethod
        def calibrated_panel_sha256(
            role: str,
            _calibration: object,
        ) -> str:
            assert role == "fit"
            return provenance.calibrated_fit_panel_sha256

    monkeypatch.setattr(
        runner,
        "_authenticated_protocols",
        lambda: (protocol, C2Protocol()),
    )
    output = tmp_path / "diagnostic.pt"
    monkeypatch.setattr(
        runner,
        "_validate_output_path",
        lambda _path: output,
    )
    code_hashes = {"runner": "a" * 64}
    monkeypatch.setattr(runner, "_code_sha256s", lambda: code_hashes)
    monkeypatch.setattr(
        runner,
        "_code_bundle_sha256",
        lambda _values: "b" * 64,
    )

    adapter = SimpleNamespace(model_fingerprint=lambda: "model")
    basis = SimpleNamespace(
        residual_width=3,
        basis_payload_sha256="c" * 64,
        source_model_sha256="d" * 64,
    )
    pre_ff3 = object()
    monkeypatch.setattr(
        runner,
        "_load_live_dependencies",
        lambda **_kwargs: (basis, adapter, pre_ff3, object(), 1e-6),
    )
    monkeypatch.setattr(
        runner,
        "_actual_model_execution",
        lambda _adapter: ("cpu", "float32"),
    )
    monkeypatch.setattr(
        runner,
        "module_state_fingerprint",
        lambda _module: "e" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_fisher_metric_weight",
        lambda _basis: torch.ones(64, dtype=torch.float64),
    )

    measured_roles: list[str] = []
    fit_probe = SimpleNamespace(
        probe=SimpleNamespace(probe_id="development_c2.fit.safe")
    )

    def measure(*, role: str, **_kwargs: object):
        measured_roles.append(role)
        return (
            (() if role == "pilot" else (fit_probe,)),
            {
                "role": role,
                "prompt_text_loaded": False,
                "token_ids_loaded": False,
            },
        )

    monkeypatch.setattr(runner, "_measure_c2_role", measure)
    pilot_metric = SimpleNamespace(state_dict=lambda: {"metric": "pilot"})
    monkeypatch.setattr(
        runner.c2,
        "_calibration_metrics",
        lambda **_kwargs: (pilot_metric,),
    )
    calibration = SimpleNamespace(
        selected_amplitude=8.0,
        artifact_sha256=provenance.calibration_sha256,
        state_dict=lambda: {
            "selected_amplitude": 8.0,
            "artifact_sha256": provenance.calibration_sha256,
        },
    )
    monkeypatch.setattr(
        runner,
        "select_global_calibration_amplitude",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setattr(
        runner.c2,
        "_fit_gauges",
        lambda *_args, **_kwargs: (
            torch.zeros(64, dtype=torch.float64),
            0.0,
            1.0,
            torch.zeros(64, dtype=torch.float64),
            torch.ones(64, dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_fit_teacher_weighted_energy",
        lambda *_args, **_kwargs: 1.0,
    )
    monkeypatch.setattr(
        runner.c2,
        "_training_contrast_pairs",
        lambda **_kwargs: ((), ()),
    )
    monkeypatch.setattr(
        runner,
        "_teacher_signal_diagnostics",
        lambda *_args, **_kwargs: {
            "minimum_teacher_delta_mse": 1.0,
            "maximum_teacher_delta_mse": 1.0,
            "minimum_teacher_jvp_mse": 1.0,
            "maximum_teacher_jvp_mse": 1.0,
        },
    )

    ordinary_splits: list[str] = []

    def ordinary(*_args: object, split: str, **_kwargs: object):
        ordinary_splits.append(split)
        return (SimpleNamespace(probe_id="ordinary.fit.00"),)

    monkeypatch.setattr(
        runner.c2,
        "_ordinary_full_width_probes",
        ordinary,
    )
    controls = SimpleNamespace(
        artifact_sha256="f" * 64,
        state_dict=lambda: {"artifact_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        runner,
        "fit_full_width_reference_controls",
        lambda **_kwargs: controls,
    )

    evaluated: list[tuple[str, str, int]] = []

    def evaluate_recipe(
        recipe,
        *,
        seed_role: str,
        seed: int,
        **_kwargs: object,
    ):
        evaluated.append((recipe.recipe_id, seed_role, seed))
        passed = recipe.recipe_id == "d1_unit_rms"
        candidate_id = f"{recipe.recipe_id}.{seed_role}"
        return runner._RecipeEvaluation(
            recipe_id=recipe.recipe_id,
            seed_role=seed_role,
            seed=seed,
            combined_pass=passed,
            row={
                "candidate_id": candidate_id,
                "_plan_state": {"artifact_sha256": "1" * 64},
            },
            plan=SimpleNamespace(artifact_sha256="1" * 64),
        )

    monkeypatch.setattr(runner, "_evaluate_recipe", evaluate_recipe)
    captured: dict[str, object] = {}

    def publish(state, report, *, output: Path):
        captured["state"] = state
        captured["report"] = report
        captured["output"] = output
        return dict(report)

    monkeypatch.setattr(runner, "_publish_artifact", publish)
    monkeypatch.setattr(
        runner,
        "load_objective_balance_diagnostic_artifact",
        lambda _path: SimpleNamespace(report=dict(captured["report"])),
    )

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("run loaded the selection-bearing C2 artifact")

    monkeypatch.setattr(torch, "load", forbidden_load)

    report = runner.run_objective_balance_diagnostic(output=output)

    assert measured_roles == ["pilot", "fit"]
    assert ordinary_splits == ["fit"]
    assert evaluated == [
        ("d0_raw_c2_control", "primary", protocol.primary_seed),
        ("d1_unit_rms", "primary", protocol.primary_seed),
        ("d1_unit_rms", "replication", protocol.replication_seed),
    ]
    assert report["selection_materialized"] is False
    assert report["selection_measured"] is False
    assert report["selection_scored"] is False
    assert report["c2_artifact_loaded"] is False
    assert report["requested_execution_device"] == "cpu"
    assert report["actual_execution_device"] == "cpu"
    assert report["requested_execution_dtype"] == "float32"
    assert report["actual_execution_dtype"] == "float32"
    assert report["authorized_fresh_c3_recipe_id"] == "d1_unit_rms"
    assert "same_authenticated_C2_fit_endpoints" in report[
        "ordinary_scoring_panel_semantics"
    ]
    assert "selection_schema" not in report[
        "ordinary_scoring_panel_semantics"
    ]
    assert report["interpretation"][
        "d3_only_pass_supports_component_attribution"
    ] is False
    assert report["interpretation"][
        "d1_changes_only_absolute_terms_proved"
    ] is False
    assert captured["output"] == output
    _assert_tensor_free(captured["report"])
