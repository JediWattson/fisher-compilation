from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_function_preserving_width_control as runner
from fisher_graph.gemma3_l3_l4_function_preserving_width_control_protocol import (
    default_function_preserving_width_control_protocol,
)
from fisher_graph.state_conditioned_contrast_fit import (
    ContrastAwareReferenceProviderPlan,
)


def _source_plan() -> ContrastAwareReferenceProviderPlan:
    source = runner.DEFAULT_SOURCE_DIAGNOSTIC
    if not source.exists():
        pytest.skip("ignored authenticated D3 source is not available")
    loaded = runner.d0d3.load_objective_balance_diagnostic_artifact(source)
    states = loaded.state["plan_states"]
    return ContrastAwareReferenceProviderPlan.from_state_dict(
        states[runner._PRIMARY_D3_ID]
    )


def _cold_models() -> tuple[object, object]:
    protocol = default_function_preserving_width_control_protocol()
    plan = _source_plan()
    rank16 = runner._new_training_model(
        modal_center=plan.modal_center,
        gain_log_center=plan.gain_log_center,
        gain_log_scale=plan.gain_log_scale,
        residual_width=plan.residual_width,
        rms_epsilon=plan.rms_epsilon,
        target_center=plan.target_center,
        target_scale=plan.target_scale,
        config=runner._executor_config(protocol.rank16_executor),
        seed=protocol.training.primary_seed,
    )
    rank64 = runner._lift_rank16_model(
        rank16,
        modal_center=plan.modal_center,
        gain_log_center=plan.gain_log_center,
        gain_log_scale=plan.gain_log_scale,
        residual_width=plan.residual_width,
        rms_epsilon=plan.rms_epsilon,
        target_center=plan.target_center,
        target_scale=plan.target_scale,
        config=runner._executor_config(protocol.rank64_executor),
        seed=protocol.training.primary_seed,
    )
    return rank16, rank64


def _synthetic_complete_payload(
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        not runner.DEFAULT_SOURCE_DIAGNOSTIC.exists()
        or not runner.DEFAULT_SOURCE_RANK64.exists()
    ):
        pytest.skip("ignored authenticated predecessor artifacts unavailable")
    protocol, objective_protocol, c2_protocol, d3_recipe = (
        runner._authenticated_declarations()
    )
    sources = runner._authenticate_sources(
        source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
        source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
        protocol=protocol,
    )
    d3_loaded = runner.d0d3.load_objective_balance_diagnostic_artifact(
        runner.DEFAULT_SOURCE_DIAGNOSTIC
    )
    rank64_loaded = runner.r64.load_rank64_capacity_control_artifact(
        runner.DEFAULT_SOURCE_RANK64
    )
    d3_row = d3_loaded.state["candidate_results"][runner._PRIMARY_D3_ID]
    rank64_source_id = "r64_d3_capacity_control.primary"
    rank64_row = rank64_loaded.state["candidate_results"][
        rank64_source_id
    ]
    rank16_plan = ContrastAwareReferenceProviderPlan.from_state_dict(
        d3_loaded.state["plan_states"][runner._PRIMARY_D3_ID]
    )
    source_rank64_plan = ContrastAwareReferenceProviderPlan.from_state_dict(
        rank64_loaded.state["plan_states"][rank64_source_id]
    )
    rank64_plan = replace(
        source_rank64_plan,
        initial_metrics=rank16_plan.initial_metrics,
        final_metrics=rank16_plan.final_metrics,
        artifact_sha256="",
    )
    plans = {
        "fp_width.primary.rank16": rank16_plan,
        "fp_width.primary.rank64": rank64_plan,
    }
    source_rows = {
        "fp_width.primary.rank16": d3_row,
        "fp_width.primary.rank64": rank64_row,
    }
    initialization = {
        "artifact_kind": "fisher_graph.initial_width_equivalence",
        "format_version": runner._FORMAT_VERSION,
        "maximum_observable_absolute_error": 0.0,
        "maximum_observable_relative_error": 0.0,
        "maximum_jvp_absolute_error": 0.0,
        "maximum_jvp_relative_error": 0.0,
        "jvp_pair_count": 32,
        "rank16_initial_metrics_sha256": (
            rank16_plan.initial_metrics.artifact_sha256
        ),
        "rank64_initial_metrics_sha256": (
            rank64_plan.initial_metrics.artifact_sha256
        ),
        "initial_parameter_bindings": (
            runner._reconstruct_initial_parameter_bindings(
                protocol,
                pair_role="primary",
            )
        ),
        "flags": {
            "observable_absolute": True,
            "observable_relative": True,
            "jvp_absolute": True,
            "jvp_relative": True,
            "initial_metrics_exact": True,
            "all_expected_jvps_compared": True,
        },
        "passed": True,
    }
    initialization["artifact_sha256"] = runner._json_sha256(
        initialization,
        domain=runner._INITIALIZATION_DOMAIN,
    )

    raw_teacher_energy = float(
        d3_loaded.report["gauge"]["raw_fit_teacher_weighted_energy"]
    )
    training_teacher_energy = float(
        d3_loaded.report["gauge"]["unit_fit_teacher_weighted_energy"]
    )
    training_teacher_signal = d3_row[
        "training_teacher_signal_diagnostics"
    ]
    ordinary_gates = runner._deferred_collision_gates(
        runner.SyntheticReferenceGates()
    )
    contrast_gates = runner.ContrastAssessmentGates()
    rows: dict[str, dict[str, object]] = {}
    for candidate_id, plan in plans.items():
        arm = candidate_id.rsplit(".", 1)[-1]
        source_row = source_rows[candidate_id]
        source_score = runner.FullWidthCandidateScore.from_state_dict(
            source_row["ordinary_score"]
        )
        score = replace(
            source_score,
            candidate_id=candidate_id,
            artifact_sha256="",
        )
        ordinary_flags = runner.r64._recompute_ordinary_gate_flags(
            score,
            ordinary_gates,
        )
        _, contrast_scores = runner.r64._validate_contrast_result_state(
            source_row["contrast_result"],
            gates=contrast_gates,
        )
        coverage = runner.r64._recompute_contrast_coverage(
            contrast_scores,
            source_row["contrast_identities"],
            required_null_candidate_pass_count=24,
        )
        contrast_result = copy.deepcopy(source_row["contrast_result"])
        ordinary_pass = all(ordinary_flags.values())
        fit_pass = (
            ordinary_pass
            and contrast_result["overall_status"] == "pass"
            and bool(coverage["every_teacher_qualified_contrast_passed"])
            and bool(coverage["all_families_cover_all_four_rank_bands"])
            and bool(coverage["required_null_contrasts_valid_and_passed"])
        )
        balance = runner.d0d3._contribution_balance_gate(
            plan,
            recipe=d3_recipe,
            gates=objective_protocol.gates,
            training_teacher_energy=training_teacher_energy,
            raw_teacher_energy=raw_teacher_energy,
            teacher_signal_diagnostics=training_teacher_signal,
        )
        rows[candidate_id] = {
            "candidate_id": candidate_id,
            "pair_role": "primary",
            "arm": arm,
            "seed": protocol.training.primary_seed,
            "latent_rank": plan.latent_rank,
            "expert_rank": plan.executor_config.expert_rank,
            "plan_sha256": plan.artifact_sha256,
            "candidate_binding_sha256": (
                score.candidate_artifact_sha256
            ),
            "source_replay_exact": runner._source_replay_exact(
                pair_role="primary",
                arm=arm,
                plan=plan,
                protocol=protocol,
            ),
            "source_sequence_comparison": (
                runner.r64._plan_sequence_comparison(
                    plan,
                    source=sources.source_d3,
                )
            ),
            "initialization_equivalence": copy.deepcopy(initialization),
            "gradient_openness": runner._gradient_audit_contract(
                protocol,
                pair_role="primary",
                arm=arm,
            ),
            "initial_training_metrics": (
                plan.initial_metrics.state_dict()
            ),
            "final_training_metrics": plan.final_metrics.state_dict(),
            "final_contribution_audit": (
                runner.audit_objective_contributions(
                    plan.final_metrics,
                    plan.objective,
                ).state_dict()
            ),
            "objective_balance_gate": balance,
            "ordinary_score": score.state_dict(),
            "contrast_result": contrast_result,
            "contrast_identities": copy.deepcopy(
                source_row["contrast_identities"]
            ),
            "contrast_coverage": coverage,
            "structural_metadata": copy.deepcopy(
                source_row["structural_metadata"]
            ),
            "accounting": asdict(plan.accounting()),
            "fit_capability_contract": {
                "ordinary_gate_count": len(ordinary_flags),
                "all_ordinary_gates_passed": ordinary_pass,
                "all_contrast_families_passed": (
                    contrast_result["overall_status"] == "pass"
                ),
                "every_qualified_contrast_passed": bool(
                    coverage[
                        "every_teacher_qualified_contrast_passed"
                    ]
                ),
                "all_four_rank_bands_covered": bool(
                    coverage["all_families_cover_all_four_rank_bands"]
                ),
                "required_null_contrasts_passed": bool(
                    coverage[
                        "required_null_contrasts_valid_and_passed"
                    ]
                ),
            },
            "fit_capability_pass": fit_pass,
        }
    validity_flags = {
        "initial_observable_and_jvp_equivalence": True,
        "gradient_open_rank64": True,
        "rank16_balance": bool(
            rows["fp_width.primary.rank16"]["objective_balance_gate"][
                "passed"
            ]
        ),
        "rank64_balance": bool(
            rows["fp_width.primary.rank64"]["objective_balance_gate"][
                "passed"
            ]
        ),
        "rank16_source_sequences": True,
        "rank64_source_sequences": True,
        "rank16_primary_replay": True,
        "initial_metrics_match": True,
        "exact_configs": True,
        "exact_training_contract": True,
    }
    assert all(validity_flags.values())
    status = runner._pair_status(
        bool(rows["fp_width.primary.rank16"]["fit_capability_pass"]),
        bool(rows["fp_width.primary.rank64"]["fit_capability_pass"]),
    )
    for row in rows.values():
        row["pair_treatment_validity"] = {
            "passed": True,
            "flags": validity_flags,
            "failure_semantics": (
                "invalid_paired_width_comparison_no_capacity_conclusion"
            ),
        }
        row["pair_comparison_status"] = status
    decision = runner._recompute_decision_from_rows(rows)
    executed = (
        "fp_width.primary.rank16",
        "fp_width.primary.rank64",
    )
    result_hashes = {
        candidate_id: runner._json_sha256(
            rows[candidate_id],
            domain=runner._RESULT_DOMAIN,
        )
        for candidate_id in executed
    }
    calibration = runner.d0d3._restore_calibration_binding(
        d3_loaded.state["calibration_state"]
    )
    controls = runner.d0d3._restore_full_width_controls(
        d3_loaded.state["controls_state"]
    )
    gauge = runner.UnitRmsFisherGauge.from_state_dict(
        d3_loaded.state["unit_rms_gauge_state"]
    )
    metric = d3_loaded.state["canonical_metric_weight"]
    code = runner._code_sha256s()
    source_manifest = d3_loaded.manifest
    manifest = {
        "schema": runner._SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "source_d3_logical_artifact_sha256": (
            protocol.sources.d3_logical_artifact_sha256
        ),
        "source_d3_tensor_file_sha256": (
            protocol.sources.d3_tensor_file_sha256
        ),
        "source_d3_report_sha256": protocol.sources.d3_report_sha256,
        "source_d3_primary_plan_sha256": (
            protocol.sources.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            protocol.sources.d3_primary_result_sha256
        ),
        "source_rank64_logical_artifact_sha256": (
            protocol.sources.rank64_logical_artifact_sha256
        ),
        "source_rank64_tensor_file_sha256": (
            protocol.sources.rank64_tensor_file_sha256
        ),
        "source_rank64_report_sha256": (
            protocol.sources.rank64_report_sha256
        ),
        "source_rank64_primary_plan_sha256": (
            protocol.sources.rank64_primary_plan_sha256
        ),
        "source_rank64_primary_result_sha256": (
            protocol.sources.rank64_primary_result_sha256
        ),
        **{
            name: source_manifest[name]
            for name in (
                "c2_protocol_sha256",
                "c2_pilot_panel_sha256",
                "c2_fit_panel_sha256",
                "c2_calibrated_fit_panel_sha256",
                "c2_calibration_sha256",
                "selected_calibration_amplitude",
                "basis_package_file_sha256",
                "basis_package_payload_sha256",
                "source_model_sha256",
                "pre_feedforward_norm_sha256",
                "canonical_metric_weight_sha256",
                "fit_data_binding_sha256",
                "unit_rms_gauge_sha256",
                "standardized_gauge_sha256",
                "controls_sha256",
                "ordinary_gates_sha256",
                "contrast_gates_sha256",
            )
        },
        "measurement_evidence_sha256": (
            protocol.training.measurement_evidence_sha256
        ),
        "requested_execution_device": protocol.execution_device,
        "requested_execution_dtype": protocol.execution_dtype,
        "actual_execution_device": protocol.execution_device,
        "actual_execution_dtype": protocol.execution_dtype,
        "executed_candidate_ids": executed,
        "candidate_plan_sha256s": {
            candidate_id: plans[candidate_id].artifact_sha256
            for candidate_id in executed
        },
        "candidate_result_sha256s": result_hashes,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_d3_source_loaded": True,
        "authenticated_rank64_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code,
        "code_bundle_sha256": runner._code_bundle_sha256(code),
        "scientific_scope": (
            "fit_only_paired_function_preserving_width_control"
        ),
    }
    assert set(manifest) == runner._MANIFEST_FIELDS
    logical = runner._json_sha256(
        manifest,
        domain=runner._ARTIFACT_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": logical,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "unit_rms_gauge_state": gauge.state_dict(),
        "canonical_metric_weight": metric,
        "controls_state": controls.state_dict(),
        "plan_states": {
            candidate_id: plans[candidate_id].state_dict()
            for candidate_id in executed
        },
        "candidate_results": rows,
    }
    gauge_report = copy.deepcopy(d3_loaded.report["gauge"])
    report_payload = {
        **manifest,
        "artifact_sha256": logical,
        "protocol": protocol.state_dict(),
        "calibration": calibration.state_dict(),
        "pilot_metrics": copy.deepcopy(
            d3_loaded.report["pilot_metrics"]
        ),
        "pilot_measurement": copy.deepcopy(
            d3_loaded.report["pilot_measurement"]
        ),
        "fit_measurement": copy.deepcopy(
            d3_loaded.report["fit_measurement"]
        ),
        "fit_provider_chart_mismatch_diagnostics": copy.deepcopy(
            d3_loaded.report[
                "fit_provider_chart_mismatch_diagnostics"
            ]
        ),
        "teacher_signal_diagnostics": copy.deepcopy(
            d3_loaded.report["teacher_signal_diagnostics"]
        ),
        "training_teacher_signal_diagnostics": copy.deepcopy(
            training_teacher_signal
        ),
        "pair_balance": copy.deepcopy(d3_row["pair_balance"]),
        "gauge": gauge_report,
        "candidate_results": [rows[value] for value in executed],
        "interpretation": runner._interpretation_from_evidence(
            rows,
            decision,
        ),
        "safety": runner._safety_contract(),
    }
    return state, report_payload


def _refresh_synthetic_outer_bindings(
    state: dict[str, object],
    report: dict[str, object],
) -> None:
    manifest = state["manifest"]
    rows = state["candidate_results"]
    assert isinstance(manifest, dict)
    assert isinstance(rows, dict)
    executed = tuple(manifest["executed_candidate_ids"])
    manifest["candidate_result_sha256s"] = {
        candidate_id: runner._json_sha256(
            rows[candidate_id],
            domain=runner._RESULT_DOMAIN,
        )
        for candidate_id in executed
    }
    logical = runner._json_sha256(
        manifest,
        domain=runner._ARTIFACT_DOMAIN,
    )
    state["artifact_sha256"] = logical
    report.update(copy.deepcopy(manifest))
    report["artifact_sha256"] = logical
    report["candidate_results"] = [
        copy.deepcopy(rows[candidate_id]) for candidate_id in executed
    ]


def _forge_contrast_pass(row: dict[str, object]) -> None:
    gates = runner.ContrastAssessmentGates()
    _, restored = runner.r64._validate_contrast_result_state(
        row["contrast_result"],
        gates=gates,
    )
    forged = []
    for score in restored:
        if score.teacher_status == "eligible_sensitivity":
            score = replace(
                score,
                candidate_contrast_l2=max(
                    1.0,
                    score.candidate_effective_noise_l2 * 1000,
                ),
                candidate_contrast_relative_error=0.0,
                candidate_direction_cosine=1.0,
                candidate_projection_gain=1.0,
                candidate_orthogonal_leakage=0.0,
                candidate_magnitude_ratio=1.0,
                candidate_gate_flags=tuple(
                    sorted(
                        (
                            ("contrast_relative_error", True),
                            ("direction_cosine", True),
                            ("projection_gain", True),
                            ("orthogonal_leakage", True),
                        )
                    )
                ),
                candidate_status="pass",
                decision_status="pass",
                reason_codes=(),
                artifact_sha256="",
            )
        elif score.teacher_status == "valid_intended_null":
            score = replace(
                score,
                candidate_null_relative_effect_upper=0.0,
                candidate_null_relative_error_upper=0.0,
                candidate_gate_flags=tuple(
                    sorted(
                        (
                            ("null_relative_effect", True),
                            ("null_relative_error", True),
                        )
                    )
                ),
                candidate_status="pass",
                decision_status="pass",
                reason_codes=(),
                artifact_sha256="",
            )
        forged.append(score)
    by_family: dict[str, list[object]] = {}
    for score in forged:
        by_family.setdefault(score.family, []).append(score)
    family_scores = tuple(
        runner.r64.contrast_assessment._family_score(
            family,
            tuple(
                sorted(
                    scores,
                    key=lambda value: value.contrast_id,
                )
            ),
            gates=gates,
        )
        for family, scores in sorted(by_family.items())
    )
    priority = runner.r64.contrast_assessment._DECISION_PRIORITY
    counts = {
        status: sum(
            value.decision_status == status
            for value in family_scores
        )
        for status in priority
    }
    result = runner.r64.contrast_assessment.ContrastAssessmentResult(
        gates_sha256=gates.artifact_sha256,
        contrast_scores=tuple(forged),
        family_scores=family_scores,
        invalid_family_count=counts["invalid"],
        teacher_null_failure_family_count=counts[
            "teacher_null_failure"
        ],
        panel_inconclusive_family_count=counts["panel_inconclusive"],
        candidate_failed_family_count=counts["candidate_fail"],
        passed_family_count=counts["pass"],
        overall_status=max(
            (value.decision_status for value in family_scores),
            key=lambda value: priority[value],
        ),
        reason_codes=tuple(
            sorted(
                {
                    f"{family.family}:{reason}"
                    for family in family_scores
                    for reason in family.reason_codes
                }
            )
        ),
    )
    coverage = runner.r64._recompute_contrast_coverage(
        forged,
        row["contrast_identities"],
        required_null_candidate_pass_count=24,
    )
    row["contrast_result"] = result.state_dict()
    row["contrast_coverage"] = coverage
    contract = row["fit_capability_contract"]
    assert isinstance(contract, dict)
    contract.update(
        {
            "all_contrast_families_passed": True,
            "every_qualified_contrast_passed": True,
            "all_four_rank_bands_covered": True,
            "required_null_contrasts_passed": True,
        }
    )
    row["fit_capability_pass"] = True


def _random_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    modal = torch.randn(2, 7, 64, generator=generator, dtype=torch.float64)
    null = torch.randn(2, 7, 1, generator=generator, dtype=torch.float64)
    rms = torch.rand(2, 7, generator=generator, dtype=torch.float64) + 0.5
    mask = torch.tensor(
        [[True] * 7, [True] * 5 + [False] * 2],
        dtype=torch.bool,
    )
    positions = torch.arange(7, dtype=torch.long).expand(2, -1)
    return modal, null, rms, mask, positions


def test_nested_lift_is_exactly_function_preserving() -> None:
    rank16, rank64 = _cold_models()
    modal, null, rms, mask, positions = _random_inputs()

    output16 = rank16.forward_standardized(  # type: ignore[attr-defined]
        modal,
        null,
        rms,
        mask,
        positions,
    )
    output64 = rank64.forward_standardized(  # type: ignore[attr-defined]
        modal,
        null,
        rms,
        mask,
        positions,
    )

    assert torch.equal(output16, output64)
    assert torch.count_nonzero(
        rank64.decoder_weight[16:]  # type: ignore[attr-defined]
    ) == 0
    assert torch.equal(
        rank64.encoder_weight,  # type: ignore[attr-defined]
        torch.eye(64, dtype=torch.float64),
    )


def test_nested_lift_jvp_is_exactly_function_preserving() -> None:
    rank16, rank64 = _cold_models()
    modal, null, rms, mask, positions = _random_inputs()
    tangents = (
        torch.ones_like(modal) * 0.1,
        torch.ones_like(null) * 0.2,
        torch.ones_like(rms) * 0.3,
    )

    def apply(model: object, m: torch.Tensor, n: torch.Tensor, r: torch.Tensor):
        return model.forward_standardized(  # type: ignore[attr-defined]
            m,
            n,
            r,
            mask,
            positions,
        )

    _, jvp16 = torch.func.jvp(
        lambda m, n, r: apply(rank16, m, n, r),
        (modal, null, rms),
        tangents,
    )
    _, jvp64 = torch.func.jvp(
        lambda m, n, r: apply(rank64, m, n, r),
        (modal, null, rms),
        tangents,
    )

    assert torch.equal(jvp16, jvp64)


def test_added_width_wakes_through_decoder_then_encoder() -> None:
    _, rank64 = _cold_models()
    modal, null, rms, mask, positions = _random_inputs()
    target = torch.randn_like(modal)
    optimizer = torch.optim.Adam(rank64.parameters(), lr=1e-3)  # type: ignore[attr-defined]
    observed: list[tuple[float, float, float]] = []

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = rank64.forward_standardized(  # type: ignore[attr-defined]
            modal,
            null,
            rms,
            mask,
            positions,
        )
        (output - target).square().mean().backward()
        decoder = float(
            rank64.decoder_weight.grad[16:].norm()  # type: ignore[attr-defined]
        )
        encoder = float(
            rank64.encoder_weight.grad[:, 16:].norm()  # type: ignore[attr-defined]
        )
        executor = float(
            rank64.executor.same_position_weight.grad[  # type: ignore[attr-defined]
                17:65,
                16:,
            ].norm()
        )
        observed.append((decoder, encoder, executor))
        optimizer.step()

    assert observed[0][0] > 1e-12
    assert observed[0][1] == 0.0
    assert observed[1][1] > 1e-12
    assert observed[1][2] > 1e-12


@pytest.mark.parametrize("pair_role", ["primary", "replication"])
def test_seeded_initial_parameter_reconstruction_matches_protocol(
    pair_role: str,
) -> None:
    protocol = default_function_preserving_width_control_protocol()

    reconstructed = runner._reconstruct_initial_parameter_bindings(
        protocol,
        pair_role=pair_role,
    )

    assert reconstructed["rank16"] == (
        protocol.initialization.hashes_for(
            seed_role=pair_role,
            arm="rank16",
        )
    )
    assert reconstructed["rank64"] == (
        protocol.initialization.hashes_for(
            seed_role=pair_role,
            arm="rank64",
        )
    )


def test_source_authentication_binds_both_predecessors() -> None:
    if (
        not runner.DEFAULT_SOURCE_DIAGNOSTIC.exists()
        or not runner.DEFAULT_SOURCE_RANK64.exists()
    ):
        pytest.skip("ignored authenticated predecessor artifacts unavailable")
    protocol = default_function_preserving_width_control_protocol()

    sources = runner._authenticate_sources(
        source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
        source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
        protocol=protocol,
    )

    assert sources.source_d3.fit_data_binding_sha256 == (
        protocol.training.fit_data_binding_sha256
    )
    assert sources.d3_primary_row["plan_sha256"] == (
        protocol.sources.d3_primary_plan_sha256
    )


def test_source_authentication_rejects_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if (
        not runner.DEFAULT_SOURCE_DIAGNOSTIC.exists()
        or not runner.DEFAULT_SOURCE_RANK64.exists()
    ):
        pytest.skip("ignored authenticated predecessor artifacts unavailable")
    protocol = default_function_preserving_width_control_protocol()
    original = runner.r64.load_rank64_capacity_control_artifact
    loaded = original(runner.DEFAULT_SOURCE_RANK64)
    tampered_manifest = dict(loaded.manifest)
    tampered_manifest["outcome"] = "rank64_two_seed_fit_capability_pass"
    tampered = SimpleNamespace(
        **{
            **{
                name: getattr(loaded, name)
                for name in loaded.__dataclass_fields__
            },
            "manifest": tampered_manifest,
        }
    )
    monkeypatch.setattr(
        runner.r64,
        "load_rank64_capacity_control_artifact",
        lambda _path: tampered,
    )

    with pytest.raises(ValueError, match="rank-64 predecessor drifted"):
        runner._authenticate_sources(
            source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
            source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
            protocol=protocol,
        )


def test_complete_synthetic_artifact_publishes_and_strictly_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, report = _synthetic_complete_payload()
    calls = {"ordinary": 0, "contrast": 0, "decision": 0}
    ordinary = runner.r64._recompute_ordinary_gate_flags
    contrast = runner.r64._validate_contrast_result_state
    decision = runner._recompute_decision_from_rows

    def observe_ordinary(*args: object, **kwargs: object) -> object:
        calls["ordinary"] += 1
        return ordinary(*args, **kwargs)  # type: ignore[arg-type]

    def observe_contrast(*args: object, **kwargs: object) -> object:
        calls["contrast"] += 1
        return contrast(*args, **kwargs)  # type: ignore[arg-type]

    def observe_decision(*args: object, **kwargs: object) -> object:
        calls["decision"] += 1
        return decision(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        runner.r64,
        "_recompute_ordinary_gate_flags",
        observe_ordinary,
    )
    monkeypatch.setattr(
        runner.r64,
        "_validate_contrast_result_state",
        observe_contrast,
    )
    monkeypatch.setattr(
        runner,
        "_recompute_decision_from_rows",
        observe_decision,
    )
    output = (tmp_path / "synthetic-valid-width.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)
    loaded = runner.load_function_preserving_width_control_artifact(
        output,
        **receipt,
    )

    assert loaded.manifest["outcome"] == "primary_both_fail"
    assert loaded.manifest["primary_treatment_valid"] is True
    assert calls == {"ordinary": 2, "contrast": 2, "decision": 1}
    assert output.is_file()
    assert output.with_suffix(".json").is_file()


def test_public_strict_loader_requires_external_trust_anchor_triple(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    output = (tmp_path / "missing-trust-anchor.pt").resolve()
    runner._publish_artifact(state, report, output=output)

    with pytest.raises(TypeError):
        runner.load_function_preserving_width_control_artifact(output)


def test_original_trust_anchor_rejects_self_rehashed_contrast_outcome(
    tmp_path: Path,
) -> None:
    valid_state, valid_report = _synthetic_complete_payload()
    forged_state, forged_report = _synthetic_complete_payload()
    output = (tmp_path / "externally-anchored-contrast.pt").resolve()
    trusted = runner._publish_artifact(
        valid_state,
        valid_report,
        output=output,
    )
    output.unlink()
    output.with_suffix(".json").unlink()
    rows = forged_state["candidate_results"]
    manifest = forged_state["manifest"]
    assert isinstance(rows, dict)
    assert isinstance(manifest, dict)
    for row in rows.values():
        _forge_contrast_pass(row)
        row["pair_comparison_status"] = "both_pass"
    decision = runner._recompute_decision_from_rows(rows)
    manifest.update(decision)
    forged_report["interpretation"] = (
        runner._interpretation_from_evidence(rows, decision)
    )
    _refresh_synthetic_outer_bindings(forged_state, forged_report)
    runner._publish_artifact(
        forged_state,
        forged_report,
        output=output,
    )

    with pytest.raises(ValueError, match="external trust anchor mismatch"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **trusted,
        )


def test_original_trust_anchor_rejects_arbitrary_nested_metadata(
    tmp_path: Path,
) -> None:
    valid_state, valid_report = _synthetic_complete_payload()
    forged_state, forged_report = _synthetic_complete_payload()
    output = (tmp_path / "externally-anchored-metadata.pt").resolve()
    trusted = runner._publish_artifact(
        valid_state,
        valid_report,
        output=output,
    )
    output.unlink()
    output.with_suffix(".json").unlink()
    rows = forged_state["candidate_results"]
    assert isinstance(rows, dict)
    metadata = rows["fp_width.primary.rank64"]["structural_metadata"]
    assert isinstance(metadata, dict)
    metadata["arbitrary_nested_content"] = {
        "prompt": "verbatim private prompt goes here"
    }
    _refresh_synthetic_outer_bindings(forged_state, forged_report)
    runner._publish_artifact(
        forged_state,
        forged_report,
        output=output,
    )

    with pytest.raises(ValueError, match="external trust anchor mismatch"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **trusted,
        )


@pytest.mark.parametrize(
    "nested_mapping",
    [
        "structural_metadata",
        "pair_treatment_validity",
        "initialization_equivalence",
        "source_sequence_comparison",
    ],
)
def test_strict_loader_rejects_prompt_bearing_nested_row_metadata(
    tmp_path: Path,
    nested_mapping: str,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    candidate_ids = ["fp_width.primary.rank64"]
    if nested_mapping in {
        "pair_treatment_validity",
        "initialization_equivalence",
    }:
        candidate_ids = [
            "fp_width.primary.rank16",
            "fp_width.primary.rank64",
        ]
    for candidate_id in candidate_ids:
        nested = rows[candidate_id][nested_mapping]
        assert isinstance(nested, dict)
        nested["audit_note"] = "verbatim private prompt"
        if nested_mapping == "initialization_equivalence":
            unhashed = dict(nested)
            unhashed.pop("artifact_sha256")
            nested["artifact_sha256"] = runner._json_sha256(
                unhashed,
                domain=runner._INITIALIZATION_DOMAIN,
            )
    _refresh_synthetic_outer_bindings(state, report)
    output = (
        tmp_path / f"contaminated-{nested_mapping}.pt"
    ).resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="(schema|metadata).*(drifted|semantics)",
    ):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_kind", "verbatim private prompt"),
        ("format_version", 2),
        ("maximum_observable_absolute_error", False),
        ("maximum_observable_relative_error", 0),
        ("maximum_jvp_absolute_error", "0.0"),
        ("jvp_pair_count", True),
        ("rank16_initial_metrics_sha256", "verbatim private prompt"),
        ("passed", 1),
    ],
)
def test_strict_loader_rejects_initialization_envelope_semantic_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    for arm in ("rank16", "rank64"):
        initialization = rows[
            f"fp_width.primary.{arm}"
        ]["initialization_equivalence"]
        assert isinstance(initialization, dict)
        initialization[field] = value
        unhashed = dict(initialization)
        unhashed.pop("artifact_sha256")
        initialization["artifact_sha256"] = runner._json_sha256(
            unhashed,
            domain=runner._INITIALIZATION_DOMAIN,
        )
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / f"initialization-{field}.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match=(
            "initialization[_ ]equivalence.*"
            "(decision|schema or semantics) drifted"
        ),
    ):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


@pytest.mark.parametrize(
    ("candidate_id", "tampered_value"),
    [
        ("fp_width.primary.rank16", False),
        ("fp_width.primary.rank64", True),
    ],
)
def test_strict_loader_rejects_self_rehashed_source_replay_tamper(
    tmp_path: Path,
    candidate_id: str,
    tampered_value: bool,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    rows[candidate_id]["source_replay_exact"] = tampered_value
    _refresh_synthetic_outer_bindings(state, report)
    output = (
        tmp_path / f"tampered-{candidate_id.rsplit('.', 1)[-1]}.pt"
    ).resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="source replay decision drifted",
    ):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_interpretation_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    interpretation = report["interpretation"]
    assert isinstance(interpretation, dict)
    interpretation["two_seed_support_implicates_outer_width"] = True
    output = (tmp_path / "tampered-interpretation.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="report safety semantics drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_initial_binding_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    for row in rows.values():
        audit = copy.deepcopy(row["initialization_equivalence"])
        audit["initial_parameter_bindings"]["rank64"][
            "executor_sha256"
        ] = "0" * 64
        audit.pop("artifact_sha256")
        audit["artifact_sha256"] = runner._json_sha256(
            audit,
            domain=runner._INITIALIZATION_DOMAIN,
        )
        row["initialization_equivalence"] = audit
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / "tampered-initial-binding.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="initial parameter binding drifted",
    ):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_gradient_norm_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_width.primary.rank64"]
    audit = copy.deepcopy(row["gradient_openness"])
    audit["extra_decoder_gradient_norm_step1"] = 123_456_789.0
    audit.pop("artifact_sha256")
    audit["artifact_sha256"] = runner._json_sha256(
        audit,
        domain=runner._INITIALIZATION_DOMAIN,
    )
    row["gradient_openness"] = audit
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / "tampered-gradient-norm.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="gradient-open flags drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_manifest_firewall_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    manifest = state["manifest"]
    assert isinstance(manifest, dict)
    manifest["actual_execution_device"] = "mps"
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / "tampered-manifest-firewall.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="manifest firewall drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_ordinary_binding_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_width.primary.rank64"]
    score = runner.FullWidthCandidateScore.from_state_dict(
        row["ordinary_score"]
    )
    row["ordinary_score"] = replace(
        score,
        candidate_artifact_sha256="0" * 64,
        artifact_sha256="",
    ).state_dict()
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / "tampered-ordinary-binding.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="ordinary/row binding drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_fit_geometry_tamper(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    plan_states = state["plan_states"]
    rows = state["candidate_results"]
    manifest = state["manifest"]
    assert isinstance(plan_states, dict)
    assert isinstance(rows, dict)
    assert isinstance(manifest, dict)
    candidate_id = "fp_width.primary.rank64"
    plan = ContrastAwareReferenceProviderPlan.from_state_dict(
        plan_states[candidate_id]
    )
    tampered = replace(
        plan,
        gain_log_center=plan.gain_log_center + 0.25,
        artifact_sha256="",
    )
    plan_states[candidate_id] = tampered.state_dict()
    rows[candidate_id]["plan_sha256"] = tampered.artifact_sha256
    manifest["candidate_plan_sha256s"][candidate_id] = (
        tampered.artifact_sha256
    )
    _refresh_synthetic_outer_bindings(state, report)
    output = (tmp_path / "tampered-fit-geometry.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="shared fit geometry drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_false_safety_wording(
    tmp_path: Path,
) -> None:
    state, report = _synthetic_complete_payload()
    safety = report["safety"]
    assert isinstance(safety, dict)
    safety["contains_regenerated_d3_equivalent_parameters"] = False
    output = (tmp_path / "tampered-safety.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="report safety semantics drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


@pytest.mark.parametrize("tamper", ["scalar", "nested_prompt"])
def test_strict_loader_rejects_report_measurement_evidence_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    state, report = _synthetic_complete_payload()
    if tamper == "scalar":
        measurement = report["pilot_measurement"]
        assert isinstance(measurement, dict)
        measurement["probe_count"] = 999_999
    else:
        report["pilot_metrics"] = [
            {"label": "verbatim private prompt goes here"}
        ]
    output = (tmp_path / f"tampered-report-{tamper}.pt").resolve()

    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="measurement evidence drifted"):
        runner.load_function_preserving_width_control_artifact(
            output,
            **receipt,
        )


@pytest.mark.parametrize(
    ("valid", "rank16", "rank64", "expected"),
    [
        (False, False, False, "invalid_primary_pair"),
        (True, False, False, "primary_both_fail"),
        (True, True, True, "primary_both_pass"),
        (True, True, False, "primary_rank16_pass_rank64_fail"),
    ],
)
def test_primary_decision_matrix_without_replication(
    valid: bool,
    rank16: bool,
    rank64: bool,
    expected: str,
) -> None:
    primary = SimpleNamespace(
        treatment_valid=valid,
        comparison_status=runner._pair_status(rank16, rank64),
    )

    decision = runner._decision(primary, None)  # type: ignore[arg-type]

    assert decision["outcome"] == expected


def test_successful_primary_requires_replication() -> None:
    primary = SimpleNamespace(
        treatment_valid=True,
        comparison_status="rank16_fail_rank64_pass",
    )

    with pytest.raises(RuntimeError, match="replication"):
        runner._decision(primary, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("valid", "status", "expected"),
    [
        (False, "rank16_fail_rank64_pass", "invalid_replication_pair"),
        (True, "rank16_fail_rank64_pass", "two_seed_outer_width_support"),
        (True, "both_fail", "replication_both_fail"),
        (True, "both_pass", "replication_both_pass"),
        (
            True,
            "rank16_pass_rank64_fail",
            "replication_rank16_pass_rank64_fail",
        ),
    ],
)
def test_replication_decision_matrix(
    valid: bool,
    status: str,
    expected: str,
) -> None:
    primary = SimpleNamespace(
        treatment_valid=True,
        comparison_status="rank16_fail_rank64_pass",
    )
    replication = SimpleNamespace(
        treatment_valid=valid,
        comparison_status=status,
    )

    decision = runner._decision(  # type: ignore[arg-type]
        primary,
        replication,
    )

    assert decision["outcome"] == expected
    assert decision["fresh_c3_authorized"] is False
    assert decision["compression_claim_authorized"] is False


def _strict_decision_pair(
    role: str,
    *,
    rank16_pass: bool,
    rank64_pass: bool,
) -> dict[str, dict[str, object]]:
    initialization = {
        "passed": True,
    }
    metrics = {"artifact_sha256": "a" * 64}
    sequences = {"passed": True}
    flags = {
        "initial_observable_and_jvp_equivalence": True,
        "gradient_open_rank64": True,
        "rank16_balance": True,
        "rank64_balance": True,
        "rank16_source_sequences": True,
        "rank64_source_sequences": True,
        "rank16_primary_replay": True,
        "initial_metrics_match": True,
        "exact_configs": True,
        "exact_training_contract": True,
    }
    validity = {
        "passed": True,
        "flags": flags,
        "failure_semantics": (
            "invalid_paired_width_comparison_no_capacity_conclusion"
        ),
    }
    status = runner._pair_status(rank16_pass, rank64_pass)

    def row(arm: str, passed: bool) -> dict[str, object]:
        return {
            "initialization_equivalence": initialization,
            "gradient_openness": {"passed": True},
            "objective_balance_gate": {"passed": True},
            "source_sequence_comparison": sequences,
            "initial_training_metrics": metrics,
            "source_replay_exact": role == "primary" and arm == "rank16",
            "pair_treatment_validity": validity,
            "fit_capability_pass": passed,
            "pair_comparison_status": status,
        }

    return {
        f"fp_width.{role}.rank16": row("rank16", rank16_pass),
        f"fp_width.{role}.rank64": row("rank64", rank64_pass),
    }


def test_strict_decision_recomputation_supports_only_same_paired_pattern() -> None:
    rows = {
        **_strict_decision_pair(
            "primary",
            rank16_pass=False,
            rank64_pass=True,
        ),
        **_strict_decision_pair(
            "replication",
            rank16_pass=False,
            rank64_pass=True,
        ),
    }

    decision = runner._recompute_decision_from_rows(rows)

    assert decision["outcome"] == "two_seed_outer_width_support"
    assert decision["compressed_width_ladder_authorized"] is True
    assert decision["fresh_c3_authorized"] is False


def test_strict_decision_recomputation_rejects_validity_tampering() -> None:
    rows = _strict_decision_pair(
        "primary",
        rank16_pass=False,
        rank64_pass=False,
    )
    rows["fp_width.primary.rank16"]["pair_treatment_validity"] = {
        "passed": False,
        "flags": rows["fp_width.primary.rank16"][
            "pair_treatment_validity"
        ]["flags"],  # type: ignore[index]
        "failure_semantics": (
            "invalid_paired_width_comparison_no_capacity_conclusion"
        ),
    }
    rows["fp_width.primary.rank64"]["pair_treatment_validity"] = (
        rows["fp_width.primary.rank16"]["pair_treatment_validity"]
    )

    with pytest.raises(ValueError, match="validity decision drifted"):
        runner._recompute_decision_from_rows(rows)


def test_strict_decision_rejects_unauthorized_replication_rows() -> None:
    rows = {
        **_strict_decision_pair(
            "primary",
            rank16_pass=False,
            rank64_pass=False,
        ),
        **_strict_decision_pair(
            "replication",
            rank16_pass=False,
            rank64_pass=True,
        ),
    }

    with pytest.raises(ValueError, match="replication pair was not authorized"):
        runner._recompute_decision_from_rows(rows)


def test_describe_does_not_load_model_or_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe must remain declaration-only")

    monkeypatch.setattr(
        runner.d0d3,
        "load_objective_balance_diagnostic_artifact",
        forbidden,
    )
    monkeypatch.setattr(
        runner.r64,
        "load_rank64_capacity_control_artifact",
        forbidden,
    )
    monkeypatch.setattr(runner, "_load_live_dependencies", forbidden)

    report = runner.describe_function_preserving_width_control()

    assert report["source_artifacts_loaded"] is False
    assert report["model_loaded"] is False
    assert report["selection_allowed"] is False
    assert report["fresh_c3_authorized"] is False


def test_parser_exposes_only_frozen_runtime_controls() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(["run"])

    assert args.device == "cpu"
    assert args.dtype == "float32"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--device", "mps"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--dtype", "bfloat16"])


def test_output_path_requires_local_runs_inside_worktree(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local-runs"):
        runner._validate_output_path(Path("paired-width.pt"))
    assert runner._validate_output_path(tmp_path / "paired-width.pt") == (
        tmp_path / "paired-width.pt"
    ).resolve()


def test_output_path_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "paired-width.pt"
    output.write_bytes(b"occupied")

    with pytest.raises(FileExistsError):
        runner._validate_output_path(output)


def test_pyproject_registers_paired_width_cli() -> None:
    contents = Path("pyproject.toml").read_text(encoding="utf-8")

    assert (
        "fisher-graph-gemma-l3-l4-function-preserving-width-dev = "
        '"fisher_graph.gemma3_l3_l4_function_preserving_width_control:main"'
        in contents
    )
