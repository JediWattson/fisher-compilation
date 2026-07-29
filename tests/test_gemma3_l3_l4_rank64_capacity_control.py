from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_rank64_capacity_control as runner
from fisher_graph.gated_executor import ResidualGatedCausalModalExecutor
from fisher_graph.gemma3_l3_l4_rank64_capacity_control_protocol import (
    default_rank64_capacity_control_protocol,
)


def _assert_tensor_free(value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("metadata contains a tensor")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_tensor_free(nested)


def _evaluation(
    *,
    role: str,
    valid: bool,
    capable: bool,
) -> runner._CapacityEvaluation:
    protocol = default_rank64_capacity_control_protocol()
    seed = (
        protocol.primary_seed
        if role == "primary"
        else protocol.replication_seed
    )
    return runner._CapacityEvaluation(
        candidate_id=f"r64_d3_capacity_control.{role}",
        seed_role=role,
        seed=seed,
        treatment_valid=valid,
        fit_capability_pass=capable,
        combined_pass=valid and capable,
        row={"candidate_id": f"r64_d3_capacity_control.{role}"},
        plan=SimpleNamespace(artifact_sha256="a" * 64),
    )


def _source_loaded(
    *,
    logical_sha256: str | None = None,
) -> SimpleNamespace:
    protocol = default_rank64_capacity_control_protocol()
    binding = protocol.source_result
    natural = tuple(f"{index + 1:064x}" for index in range(48))
    balanced = tuple(f"{index + 101:064x}" for index in range(56))
    sequences = {
        "fit_batch_sha256s": tuple(
            f"{index + 201:064x}" for index in range(8)
        ),
        "fit_batch_content_sha256s": tuple(
            f"{index + 301:064x}" for index in range(8)
        ),
        "fit_indexed_batch_sha256s": tuple(
            f"{index + 401:064x}" for index in range(8)
        ),
        "fit_endpoint_sha256s": tuple(
            f"{index + 501:064x}" for index in range(80)
        ),
        "fit_pair_sha256s": balanced,
    }
    shared_names = (
        "c2_protocol_sha256",
        "c2_pilot_panel_sha256",
        "c2_fit_panel_sha256",
        "c2_calibrated_fit_panel_sha256",
        "c2_calibration_sha256",
        "basis_package_file_sha256",
        "basis_package_payload_sha256",
        "source_model_sha256",
        "pre_feedforward_norm_sha256",
        "canonical_metric_weight_sha256",
        "unit_rms_gauge_sha256",
        "standardized_gauge_sha256",
        "controls_sha256",
        "ordinary_gates_sha256",
        "contrast_gates_sha256",
    )
    shared = {
        name: f"{index + 1001:064x}"
        for index, name in enumerate(shared_names)
    }
    shared["selected_calibration_amplitude"] = 8.0
    candidate_id = runner._SOURCE_D3_CANDIDATE_ID
    manifest = {
        "protocol_sha256": binding.protocol_sha256,
        "code_bundle_sha256": binding.code_bundle_sha256,
        "outcome": "no_primary_treatment_passed_fit_gates",
        "authorized_fresh_c3_recipe_id": None,
        "candidate_plan_sha256s": {
            candidate_id: binding.d3_primary_plan_sha256
        },
        "candidate_result_sha256s": {
            candidate_id: binding.d3_primary_result_sha256
        },
        **shared,
    }
    row = {
        "recipe_sha256": binding.d3_recipe_sha256,
        "plan_sha256": binding.d3_primary_plan_sha256,
        "fit_data_binding_sha256": binding.fit_data_binding_sha256,
        "latent_rank": protocol.baseline_latent_rank,
        "seed": protocol.primary_seed,
        "seed_role": "primary",
        "advancement_fit_gate_pass": False,
        "pair_balance": {
            "natural_pair_sha256s": natural,
            "balanced_pair_sha256s": balanced,
        },
    }
    # Sentinel source weights prove the returned binding has no parameter
    # field and cannot become a warm-start input.
    plan_state = {
        **sequences,
        "encoder_weight": torch.ones(64, 16),
        "decoder_weight": torch.ones(16, 64),
    }
    return SimpleNamespace(
        artifact_sha256=(
            binding.logical_artifact_sha256
            if logical_sha256 is None
            else logical_sha256
        ),
        tensor_file_sha256=binding.tensor_sha256,
        report_sha256=binding.report_sha256,
        manifest=manifest,
        state={
            "candidate_results": {candidate_id: row},
            "plan_states": {candidate_id: plan_state},
        },
    )


def test_parser_exposes_only_frozen_rank64_controls() -> None:
    parser = runner.build_parser()
    describe = parser.parse_args(["describe"])
    run = parser.parse_args(["run"])

    assert describe.command == "describe"
    assert run.command == "run"
    assert run.source_diagnostic == runner.DEFAULT_SOURCE_DIAGNOSTIC
    assert run.output == runner.DEFAULT_OUTPUT
    assert run.device == "cpu"
    assert run.dtype == "float32"
    for forbidden in ("--rank", "--steps", "--seed", "--expert-rank", "--force"):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", forbidden, "1"])


def test_describe_opens_no_model_or_source_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe opened live state")

    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)
    monkeypatch.setattr(runner, "_authenticate_source_d3", forbidden)

    report = runner.describe_rank64_capacity_control()

    assert report["baseline_latent_rank"] == 16
    assert report["latent_rank"] == 64
    assert report["expert_rank"] == 16
    assert report["source_diagnostic_loading_in_describe"] is False
    assert report["selection_materialized"] is False
    assert report["fresh_c3_authorized"] is False
    assert report["compression_claim_authorized"] is False
    _assert_tensor_free(report)


def test_nonfrozen_execution_rejected_before_source_or_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution mismatch opened live state")

    monkeypatch.setattr(runner, "_authenticate_source_d3", forbidden)
    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)

    with pytest.raises(ValueError, match="frozen to cpu/float32"):
        runner.run_rank64_capacity_control(device_name="mps")
    with pytest.raises(ValueError, match="frozen to cpu/float32"):
        runner.run_rank64_capacity_control(dtype="bfloat16")


def test_executor_changes_only_outer_width_and_keeps_causal_core() -> None:
    protocol = default_rank64_capacity_control_protocol()
    config = runner._executor_config(protocol)

    assert config.input_modes == 66
    assert config.output_modes == 64
    assert config.expert_count == 2
    assert config.expert_rank == 16
    assert config.router_width == 16
    assert config.same_position_skip is False
    assert config.max_positive_lag is None
    assert config.router_activation == "tanh"
    assert config.source_normalized_routing is True
    with pytest.raises(ValueError, match="frozen ladder"):
        runner.c2._executor_config(64)


def test_provider_binding_explicitly_binds_rank64_and_cold_start() -> None:
    protocol = default_rank64_capacity_control_protocol()
    metric = torch.ones(64, dtype=torch.float64)
    binding = runner._provider_binding_sha256(
        protocol=protocol,
        c2_protocol_sha256="a" * 64,
        calibration_sha256="b" * 64,
        basis_payload_sha256="c" * 64,
        source_model_sha256="d" * 64,
        norm_sha256="e" * 64,
        training_metric_weight=metric,
        target_center=torch.zeros(64, dtype=torch.float64),
        target_scale=torch.ones(64, dtype=torch.float64),
    )
    source_d3 = runner.d0d3._provider_binding_sha256(
        diagnostic_protocol_sha256=(
            protocol.source_result.protocol_sha256
        ),
        recipe=runner.default_objective_balance_diagnostic_protocol().recipe(
            protocol.training.recipe_id
        ),
        c2_protocol_sha256="a" * 64,
        calibration_sha256="b" * 64,
        basis_payload_sha256="c" * 64,
        source_model_sha256="d" * 64,
        norm_sha256="e" * 64,
        training_metric_weight=metric,
        target_center=torch.zeros(64, dtype=torch.float64),
        target_scale=torch.ones(64, dtype=torch.float64),
    )

    assert len(binding) == 64
    assert binding != source_d3


def test_source_authentication_discards_source_provider_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_rank64_capacity_control_protocol()
    loaded = _source_loaded()
    monkeypatch.setattr(
        runner.d0d3,
        "load_objective_balance_diagnostic_artifact",
        lambda _path: loaded,
    )
    monkeypatch.setattr(
        runner,
        "_validate_source_sequence_anchors",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "source_replay_binding_sha256",
        lambda **_kwargs: protocol.source_result.d3_source_replay_binding_sha256,
    )

    source = runner._authenticate_source_d3(
        Path("source.pt"),
        protocol=protocol,
    )

    assert source.fit_data_binding_sha256 == (
        protocol.training.fit_data_binding_sha256
    )
    assert source.source_plan_parameters_used is False
    assert not hasattr(source, "encoder_weight")
    assert not hasattr(source, "decoder_weight")
    _assert_tensor_free(source.sequence_sha256s)
    _assert_tensor_free(source.shared_manifest)


def test_source_authentication_rejects_any_bound_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_rank64_capacity_control_protocol()
    monkeypatch.setattr(
        runner.d0d3,
        "load_objective_balance_diagnostic_artifact",
        lambda _path: _source_loaded(logical_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="source drifted"):
        runner._authenticate_source_d3(
            Path("source.pt"),
            protocol=protocol,
        )


def test_schedule_replicates_only_a_complete_primary_pass() -> None:
    protocol = default_rank64_capacity_control_protocol()
    calls: list[str] = []

    def failing(role: str, _seed: int) -> runner._CapacityEvaluation:
        calls.append(role)
        return _evaluation(role=role, valid=True, capable=False)

    primary, replication = runner._execute_capacity_schedule(
        protocol,
        evaluate=failing,
    )
    assert primary.fit_capability_pass is False
    assert replication is None
    assert calls == ["primary"]

    calls.clear()

    def passing(role: str, _seed: int) -> runner._CapacityEvaluation:
        calls.append(role)
        return _evaluation(role=role, valid=True, capable=True)

    primary, replication = runner._execute_capacity_schedule(
        protocol,
        evaluate=passing,
    )
    assert primary.combined_pass
    assert replication is not None and replication.combined_pass
    assert calls == ["primary", "replication"]


@pytest.mark.parametrize(
    ("primary", "replication", "outcome"),
    (
        (
            _evaluation(role="primary", valid=False, capable=True),
            None,
            "invalid_rank_comparison_primary_treatment_validity_failed",
        ),
        (
            _evaluation(role="primary", valid=True, capable=False),
            None,
            "rank64_primary_fit_capability_failed",
        ),
        (
            _evaluation(role="primary", valid=True, capable=True),
            _evaluation(role="replication", valid=False, capable=True),
            "invalid_rank_comparison_replication_treatment_validity_failed",
        ),
        (
            _evaluation(role="primary", valid=True, capable=True),
            _evaluation(role="replication", valid=True, capable=False),
            "rank64_replication_fit_capability_failed",
        ),
        (
            _evaluation(role="primary", valid=True, capable=True),
            _evaluation(role="replication", valid=True, capable=True),
            "rank64_two_seed_fit_capability_pass",
        ),
    ),
)
def test_decision_separates_validity_capacity_and_authority(
    primary: runner._CapacityEvaluation,
    replication: runner._CapacityEvaluation | None,
    outcome: str,
) -> None:
    decision = runner._capacity_decision(primary, replication)

    assert decision["outcome"] == outcome
    assert decision["fresh_c3_authorized"] is False
    assert decision["compression_claim_authorized"] is False
    assert decision[
        "compressed_width_ladder_preregistration_supported"
    ] is (outcome == "rank64_two_seed_fit_capability_pass")


def test_post_publish_authentication_failure_removes_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rejected.pt"

    monkeypatch.setattr(
        runner,
        "load_rank64_capacity_control_artifact",
        lambda _path: (_ for _ in ()).throw(ValueError("rejected")),
    )

    with pytest.raises(ValueError, match="rejected"):
        runner._publish_and_authenticate_artifact(
            {"safe_weight": torch.ones(1)},
            {"schema": "test"},
            output=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_output_path_atomic_no_overwrite_and_artifact_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.setattr(runner, "find_git_worktree", lambda _path: worktree)
    allowed = worktree / ".local-runs" / "capacity.pt"
    assert runner._validate_output_path(allowed) == allowed.resolve()
    with pytest.raises(ValueError, match="ignored local-runs"):
        runner._validate_output_path(worktree / "capacity.pt")
    with pytest.raises(ValueError, match=r"\.pt suffix"):
        runner._validate_output_path(
            worktree / ".local-runs" / "capacity.json"
        )

    output = tmp_path / "published.pt"
    runner._publish_artifact(
        {
            "provider_weight": torch.ones(2),
            "candidate": {
                "accounting": {"target_modes": runner._MODAL_WIDTH}
            },
        },
        {
            "schema": "safe-test",
            "candidate": {
                "accounting": {"target_modes": runner._MODAL_WIDTH}
            },
        },
        output=output,
    )
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._publish_artifact({}, {}, output=output)
    with pytest.raises(ValueError, match="forbidden"):
        runner._publish_artifact(
            {"teacher_midpoint_jvp": torch.ones(1)},
            {},
            output=tmp_path / "unsafe.pt",
        )


def test_loader_requests_weights_only_before_rejecting_invalid_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "invalid.pt"
    torch.save({"wrong": torch.ones(1)}, output)
    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    original_load = torch.load
    calls: list[object] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    with pytest.raises(ValueError, match="tensor fields"):
        runner.load_rank64_capacity_control_artifact(output)
    assert calls == [True]


def test_runner_reuses_selection_firewall_and_never_allows_selection_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    def measure(*, role: str, **_kwargs: object) -> object:
        delegated.append(role)
        return (), {"role": role}

    monkeypatch.setattr(runner.c2, "_measure_role", measure)
    assert runner.d0d3._measure_c2_role(role="pilot")[1]["role"] == "pilot"
    assert runner.d0d3._measure_c2_role(role="fit")[1]["role"] == "fit"
    with pytest.raises(PermissionError, match="forbids C2 selection"):
        runner.d0d3._measure_c2_role(role="selection")
    assert delegated == ["pilot", "fit"]


def test_pyproject_registers_rank64_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "fisher-graph-gemma-l3-l4-rank64-capacity-dev = "
        '"fisher_graph.gemma3_l3_l4_rank64_capacity_control:main"'
    ) in pyproject


def test_live_source_artifact_still_authenticates_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / runner.DEFAULT_SOURCE_DIAGNOSTIC
    if not source.exists():
        pytest.skip("ignored objective-balance artifact is not available")

    loaded = runner.d0d3.load_objective_balance_diagnostic_artifact(source)
    protocol = default_rank64_capacity_control_protocol()
    bindings = runner._authenticate_source_d3(
        source,
        protocol=protocol,
    )

    assert loaded.artifact_sha256 == (
        protocol.source_result.logical_artifact_sha256
    )
    assert bindings.source_plan_parameters_used is False
    assert len(bindings.natural_pair_sha256s) == 48
    assert len(bindings.balanced_pair_sha256s) == 56


def test_synthetic_complete_artifact_publishes_and_strictly_reloads_when_available(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / runner.DEFAULT_SOURCE_DIAGNOSTIC
    if not source_path.exists():
        pytest.skip("ignored objective-balance artifact is not available")

    source = runner.d0d3.load_objective_balance_diagnostic_artifact(
        source_path
    )
    protocol = default_rank64_capacity_control_protocol()
    source_result = protocol.source_result
    source_bindings = runner._authenticate_source_d3(
        source_path,
        protocol=protocol,
    )
    objective_protocol = (
        runner.default_objective_balance_diagnostic_protocol()
    )
    c2_protocol = runner.default_contrast_provider_development_protocol()
    d3_recipe = objective_protocol.recipe(protocol.training.recipe_id)
    source_row = source.state["candidate_results"][
        runner._SOURCE_D3_CANDIDATE_ID
    ]
    source_plan = runner.ContrastAwareReferenceProviderPlan.from_state_dict(
        source.state["plan_states"][runner._SOURCE_D3_CANDIDATE_ID]
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.primary_seed)
        executor = ResidualGatedCausalModalExecutor(
            runner._executor_config(protocol),
            dtype=torch.float64,
        )
    synthetic_plan = runner.ContrastAwareReferenceProviderPlan(
        modal_center=source_plan.modal_center,
        gain_log_center=source_plan.gain_log_center,
        gain_log_scale=source_plan.gain_log_scale,
        residual_width=source_plan.residual_width,
        rms_epsilon=source_plan.rms_epsilon,
        target_center=source_plan.target_center,
        target_scale=source_plan.target_scale,
        encoder_weight=torch.eye(64, dtype=torch.float64),
        executor_artifact=executor.artifact_state_dict(),
        decoder_weight=torch.eye(64, dtype=torch.float64),
        fisher_metric_weight=source_plan.fisher_metric_weight,
        fisher_metric_supplied=True,
        synthetic_binding_sha256=source_result.fit_data_binding_sha256,
        fit_batch_sha256s=source_plan.fit_batch_sha256s,
        fit_batch_content_sha256s=(
            source_plan.fit_batch_content_sha256s
        ),
        fit_indexed_batch_sha256s=(
            source_plan.fit_indexed_batch_sha256s
        ),
        fit_endpoint_sha256s=source_plan.fit_endpoint_sha256s,
        fit_pair_sha256s=source_plan.fit_pair_sha256s,
        objective=runner._objective(protocol.training),
        training_steps=protocol.training.steps,
        learning_rate=protocol.training.learning_rate,
        seed=protocol.primary_seed,
        initial_metrics=source_plan.initial_metrics,
        final_metrics=source_plan.final_metrics,
    )
    synthetic_plan_state = runner.d0d3._round_trip_plan_state(
        synthetic_plan
    )
    candidate_id = "r64_d3_capacity_control.primary"
    candidate_binding_sha256 = "f" * 64
    source_ordinary_score = runner.FullWidthCandidateScore.from_state_dict(
        source_row["ordinary_score"]
    )
    ordinary_score = replace(
        source_ordinary_score,
        candidate_id=candidate_id,
        candidate_artifact_sha256=candidate_binding_sha256,
        stored_scalar_count=(
            synthetic_plan.accounting().total_stored_scalar_count
        ),
        artifact_sha256="",
    )
    coverage = copy.deepcopy(source_row["contrast_coverage"])
    contrast_result = copy.deepcopy(source_row["contrast_result"])
    ordinary_flags = ordinary_score.gate_flags.state_dict()
    ordinary_pass = (
        len(ordinary_flags)
        == objective_protocol.gates.required_ordinary_gate_count
        and all(ordinary_flags.values())
    )
    capability_contract = {
        "ordinary_gate_count": len(ordinary_flags),
        "required_ordinary_gate_count": (
            objective_protocol.gates.required_ordinary_gate_count
        ),
        "all_ordinary_gates_passed": ordinary_pass,
        "every_eligible_sensitivity_passed": bool(
            coverage["every_teacher_qualified_contrast_passed"]
        ),
        "all_contrast_families_formally_passed": (
            contrast_result["overall_status"] == "pass"
        ),
        "all_families_cover_all_four_rank_bands": bool(
            coverage["all_families_cover_all_four_rank_bands"]
        ),
        "required_null_contrasts_passed": bool(
            coverage["required_null_contrasts_valid_and_passed"]
        ),
    }
    fit_capability_pass = all(
        (
            capability_contract["all_ordinary_gates_passed"],
            capability_contract["every_eligible_sensitivity_passed"],
            capability_contract[
                "all_contrast_families_formally_passed"
            ],
            capability_contract[
                "all_families_cover_all_four_rank_bands"
            ],
            capability_contract["required_null_contrasts_passed"],
        )
    )
    assert fit_capability_pass is False
    source_comparison = runner._plan_sequence_comparison(
        synthetic_plan,
        source=source_bindings,
    )
    treatment_flags = {
        "objective_contribution_balance": bool(
            source_row["objective_balance_gate"]["passed"]
        ),
        "exact_source_fit_sequences": bool(source_comparison["passed"]),
        "exact_rank64_executor_config": True,
        "exact_d3_objective": True,
        "exact_steps": True,
        "exact_learning_rate": True,
        "exact_seed": True,
        "exact_unit_rms_training_metric": True,
        "exact_fit_data_binding": True,
        "cold_start_source_weights_unused": True,
    }
    assert all(treatment_flags.values())

    calibration_state = source.state["calibration_state"]
    gauge_state = source.state["unit_rms_gauge_state"]
    controls_state = source.state["controls_state"]
    calibration = runner.d0d3._restore_calibration_binding(
        calibration_state
    )
    gauge = runner.UnitRmsFisherGauge.from_state_dict(gauge_state)
    controls = runner.d0d3._restore_full_width_controls(controls_state)
    raw_metric_weight = source.state["canonical_metric_weight"]
    provider_binding_sha256 = runner._provider_binding_sha256(
        protocol=protocol,
        c2_protocol_sha256=c2_protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        basis_payload_sha256=str(
            source_bindings.shared_manifest[
                "basis_package_payload_sha256"
            ]
        ),
        source_model_sha256=str(
            source_bindings.shared_manifest["source_model_sha256"]
        ),
        norm_sha256=str(
            source_bindings.shared_manifest[
                "pre_feedforward_norm_sha256"
            ]
        ),
        training_metric_weight=synthetic_plan.fisher_metric_weight,
        target_center=synthetic_plan.target_center,
        target_scale=synthetic_plan.target_scale,
    )
    candidate_row = {
        "candidate_id": candidate_id,
        "treatment_id": "r64_d3_capacity_control",
        "source_recipe_id": d3_recipe.recipe_id,
        "source_recipe_sha256": d3_recipe.artifact_sha256,
        "training_sha256": protocol.training.artifact_sha256,
        "seed_role": "primary",
        "seed": protocol.primary_seed,
        "baseline_latent_rank": protocol.baseline_latent_rank,
        "latent_rank": protocol.latent_rank,
        "expert_rank": protocol.executor.expert_rank,
        "cold_start": True,
        "source_plan_parameters_used_for_initialization": False,
        "training_metric": protocol.training.training_metric,
        "training_metric_weight_sha256": runner._tensor_sha256(
            synthetic_plan.fisher_metric_weight
        ),
        "canonical_scoring_metric_weight_sha256": runner._tensor_sha256(
            raw_metric_weight
        ),
        "pair_balance": copy.deepcopy(source_row["pair_balance"]),
        "training_teacher_signal_diagnostics": copy.deepcopy(
            source_row["training_teacher_signal_diagnostics"]
        ),
        "provider_binding_sha256": provider_binding_sha256,
        "fit_data_binding_sha256": (
            source_result.fit_data_binding_sha256
        ),
        "fit_data_binding_recipe_independent": True,
        "source_fit_sequence_comparison": source_comparison,
        "plan_sha256": synthetic_plan.artifact_sha256,
        "plan_round_trip_passed": True,
        "accounting": asdict(synthetic_plan.accounting()),
        "execution_accounting": copy.deepcopy(
            source_row["execution_accounting"]
        ),
        "initial_training_metrics": (
            synthetic_plan.initial_metrics.state_dict()
        ),
        "final_training_metrics": (
            synthetic_plan.final_metrics.state_dict()
        ),
        "final_contribution_audit": (
            runner.audit_objective_contributions(
                synthetic_plan.final_metrics,
                synthetic_plan.objective,
            ).state_dict()
        ),
        "objective_balance_gate": copy.deepcopy(
            source_row["objective_balance_gate"]
        ),
        "treatment_validity": {
            "passed": True,
            "flags": treatment_flags,
            "failure_semantics": (
                "invalid_rank_comparison_no_capacity_conclusion"
            ),
        },
        "fit_capability_contract": capability_contract,
        "ordinary_score": ordinary_score.state_dict(),
        "contrast_result": contrast_result,
        "contrast_coverage": coverage,
        "contrast_identities": copy.deepcopy(
            source_row["contrast_identities"]
        ),
        "structural_metadata": copy.deepcopy(
            source_row["structural_metadata"]
        ),
        "mode_packing": runner.c2._mode_packing_diagnostics(
            synthetic_plan
        ),
        "fit_capability_pass": False,
        "combined_pass": False,
        "failure_reasons": tuple(source_row["failure_reasons"]),
        "candidate_binding_sha256": candidate_binding_sha256,
        "contains_raw_fit_targets": False,
        "contains_teacher_jvp_tensors": False,
        "contains_provider_chart_tensors": False,
    }
    assert set(candidate_row) == runner._CANDIDATE_ROW_FIELDS
    candidate_result_sha256 = runner._json_sha256(
        candidate_row,
        domain=runner._ARTIFACT_DOMAIN,
    )
    primary = runner._CapacityEvaluation(
        candidate_id=candidate_id,
        seed_role="primary",
        seed=protocol.primary_seed,
        treatment_valid=True,
        fit_capability_pass=False,
        combined_pass=False,
        row=candidate_row,
        plan=synthetic_plan,
    )
    decision = runner._capacity_decision(primary, None)
    code_sha256s = runner._code_sha256s()
    shared = dict(source_bindings.shared_manifest)
    manifest = {
        "schema": runner._SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "source_objective_protocol_sha256": (
            source_result.protocol_sha256
        ),
        "source_objective_result_binding_sha256": (
            source_result.artifact_sha256
        ),
        "source_objective_logical_artifact_sha256": (
            source_result.logical_artifact_sha256
        ),
        "source_objective_tensor_file_sha256": (
            source_result.tensor_sha256
        ),
        "source_objective_report_sha256": source_result.report_sha256,
        "source_objective_code_bundle_sha256": (
            source_result.code_bundle_sha256
        ),
        "source_d3_recipe_sha256": source_result.d3_recipe_sha256,
        "source_d3_primary_plan_sha256": (
            source_result.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            source_result.d3_primary_result_sha256
        ),
        "source_d3_fit_data_binding_sha256": (
            source_result.fit_data_binding_sha256
        ),
        **shared,
        "requested_execution_device": protocol.execution_device,
        "requested_execution_dtype": protocol.execution_dtype,
        "actual_execution_device": protocol.execution_device,
        "actual_execution_dtype": protocol.execution_dtype,
        "fit_data_binding_sha256": (
            source_result.fit_data_binding_sha256
        ),
        "baseline_latent_rank": protocol.baseline_latent_rank,
        "latent_rank": protocol.latent_rank,
        "expert_rank": protocol.executor.expert_rank,
        "controlled_change": "outer_latent_rank_16_to_64_only",
        "executed_candidate_ids": (candidate_id,),
        "candidate_plan_sha256s": {
            candidate_id: synthetic_plan.artifact_sha256
        },
        "candidate_result_sha256s": {
            candidate_id: candidate_result_sha256
        },
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "selection_data_changed_training": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_source_result_artifact_loaded": True,
        "source_d3_parameters_used_for_initialization": False,
        "cold_start": True,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": runner._code_bundle_sha256(
            code_sha256s
        ),
        "scientific_scope": (
            "fit_only_rank64_capacity_control_not_compression_or_"
            "generalization"
        ),
    }
    assert set(manifest) == runner._MANIFEST_FIELDS
    artifact_sha256 = runner._json_sha256(
        manifest,
        domain=runner._ARTIFACT_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": artifact_sha256,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "unit_rms_gauge_state": gauge.state_dict(),
        "canonical_metric_weight": raw_metric_weight,
        "controls_state": controls.state_dict(),
        "plan_states": {candidate_id: synthetic_plan_state},
        "candidate_results": {candidate_id: candidate_row},
    }
    report_payload = {
        **manifest,
        "artifact_sha256": artifact_sha256,
        "protocol": protocol.state_dict(),
        "calibration": calibration.state_dict(),
        "pilot_metrics": copy.deepcopy(source.report["pilot_metrics"]),
        "pilot_measurement": copy.deepcopy(
            source.report["pilot_measurement"]
        ),
        "fit_measurement": copy.deepcopy(
            source.report["fit_measurement"]
        ),
        "fit_provider_chart_mismatch_diagnostics": copy.deepcopy(
            source.report["fit_provider_chart_mismatch_diagnostics"]
        ),
        "teacher_signal_diagnostics": copy.deepcopy(
            source.report["teacher_signal_diagnostics"]
        ),
        "gauge": copy.deepcopy(source.report["gauge"]),
        "source_d3_binding": {
            "source_result": source_result.state_dict(),
            "source_fit_sequence_sha256s": dict(
                source_bindings.sequence_sha256s
            ),
            "source_natural_pair_sha256s": (
                source_bindings.natural_pair_sha256s
            ),
            "source_balanced_pair_sha256s": (
                source_bindings.balanced_pair_sha256s
            ),
            "source_shared_manifest": shared,
            "source_plan_parameters_used": False,
        },
        "candidate_results": [candidate_row],
        "interpretation": {
            "fit_side_only": True,
            "held_out_selection_evidence": False,
            "rank64_is_capacity_oracle_not_compression": True,
            "two_seed_pass_implicates_outer_packing_bottleneck": True,
            "two_seed_pass_proves_rank16_is_only_bottleneck": False,
            "two_seed_pass_opens_only_separate_width_ladder": True,
            "valid_failure_leaves_executor_objective_optimization_entangled": (
                True
            ),
            "rank64_failure_proves_insufficient_capacity": False,
            "fresh_c3_authorized": False,
            "natural_prompt_fidelity_claim": False,
            "whole_model_replacement_claim": False,
            "wall_clock_speed_claim": False,
            "whole_model_compression_claim": False,
            "provider_fit_numeric_dtype": "torch.float64",
            "live_measurement_device": protocol.execution_device,
            "live_measurement_dtype": protocol.execution_dtype,
        },
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_rank64_provider_parameters": True,
            "contains_source_d3_provider_parameters": False,
            "contains_raw_teacher_targets": False,
            "contains_teacher_jvp_tensors": False,
            "contains_provider_chart_jvp_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_c2_selection_data": False,
            "committable": False,
        },
    }
    output = tmp_path / "synthetic-rank64.pt"
    loaded = runner._publish_and_authenticate_artifact(
        state,
        report_payload,
        output=output,
    )

    assert loaded.artifact_sha256 == artifact_sha256
    assert loaded.manifest["outcome"] == (
        "rank64_primary_fit_capability_failed"
    )
    restored_plan = runner.ContrastAwareReferenceProviderPlan.from_state_dict(
        loaded.state["plan_states"][candidate_id]
    )
    assert restored_plan.latent_rank == 64
    assert output.is_file()
    assert output.with_suffix(".json").is_file()


def test_contrast_aggregate_rejects_self_consistent_outcome_flip_when_available(
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / runner.DEFAULT_SOURCE_DIAGNOSTIC
    if not source.exists():
        pytest.skip("ignored objective-balance artifact is not available")
    loaded = runner.d0d3.load_objective_balance_diagnostic_artifact(source)
    row = loaded.state["candidate_results"][
        runner._SOURCE_D3_CANDIDATE_ID
    ]
    contrast = copy.deepcopy(row["contrast_result"])
    runner._validate_contrast_result_state(
        contrast,
        gates=runner.ContrastAssessmentGates(),
    )

    contrast["overall_status"] = "pass"
    payload = dict(contrast)
    payload.pop("artifact_sha256")
    payload.pop("contrast_scores")
    payload.pop("family_scores")
    contrast["artifact_sha256"] = runner.contrast_assessment._digest(
        payload,
        domain=runner.contrast_assessment._ASSESSMENT_DOMAIN,
    )
    with pytest.raises(ValueError, match="aggregation drifted"):
        runner._validate_contrast_result_state(
            contrast,
            gates=runner.ContrastAssessmentGates(),
        )


def test_schedule_rejects_self_inconsistent_combined_flag() -> None:
    protocol = default_rank64_capacity_control_protocol()
    inconsistent = _evaluation(role="primary", valid=True, capable=False)
    object.__setattr__(inconsistent, "combined_pass", True)

    with pytest.raises(RuntimeError, match="identity drifted"):
        runner._execute_capacity_schedule(
            protocol,
            evaluate=lambda _role, _seed: inconsistent,
        )
