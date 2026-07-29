from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_function_preserving_expert_count_control as runner
from fisher_graph.gated_executor import ResidualGatedCausalModalExecutor
from fisher_graph.state_conditioned_contrast_fit import (
    ContrastAwareReferenceProviderPlan,
)


def _require_sources() -> None:
    for path in (
        runner.DEFAULT_SOURCE_EXPERT_RANK,
        runner.DEFAULT_SOURCE_DIAGNOSTIC,
        runner.DEFAULT_SOURCE_RANK64,
    ):
        if not path.exists():
            pytest.skip(f"ignored authenticated source is unavailable: {path}")


def _not_applicable_parity(pair_role: str) -> dict[str, object]:
    state: dict[str, object] = {
        "artifact_kind": "fisher_graph.expert_count_wrapper_concat_parity",
        "format_version": runner._FORMAT_VERSION,
        "pair_role": pair_role,
        "stage": "not_applicable_control",
        "applicable": False,
        "passed": True,
    }
    state["artifact_sha256"] = runner._json_sha256(
        state,
        domain=runner._AUDIT_DOMAIN,
    )
    return state


def _treatment_parity(
    plan: ContrastAwareReferenceProviderPlan,
    *,
    pair_role: str,
) -> dict[str, object]:
    state: dict[str, object] = {
        "artifact_kind": "fisher_graph.expert_count_wrapper_concat_parity",
        "format_version": runner._FORMAT_VERSION,
        "pair_role": pair_role,
        "stage": "post_fit",
        "maximum_output_absolute_error": 0.0,
        "maximum_output_relative_error": 0.0,
        "maximum_jvp_absolute_error": 0.0,
        "maximum_jvp_relative_error": 0.0,
        "weighted_total_absolute_error": 0.0,
        "jvp_pair_count": 32,
        "concatenated_metrics_sha256": (
            plan.final_metrics.artifact_sha256
        ),
        "concatenated_executor_sha256": (
            runner.contrast_fit._executor_artifact_sha256(
                plan.executor_artifact
            )
        ),
        "flags": {
            "output_absolute": True,
            "output_relative": True,
            "jvp_absolute": True,
            "jvp_relative": True,
            "weighted_total_absolute": True,
            "all_expected_jvps_compared": True,
        },
        "passed": True,
    }
    state["artifact_sha256"] = runner._json_sha256(
        state,
        domain=runner._AUDIT_DOMAIN,
    )
    return state


def _lift_standard_e2_executor(
    plan: ContrastAwareReferenceProviderPlan,
    *,
    target_config: object,
) -> ResidualGatedCausalModalExecutor:
    source = ResidualGatedCausalModalExecutor.from_artifact_state_dict(
        plan.executor_artifact
    )
    target = ResidualGatedCausalModalExecutor(
        target_config,  # type: ignore[arg-type]
        dtype=source.dtype,
        device=source.device,
    )
    source_state = source.state_dict()
    target_state = target.state_dict()
    with torch.no_grad():
        for name, value in target_state.items():
            parent = source_state[name]
            if name == "expert_input_weight":
                value[:2].copy_(parent)
                value[2:].copy_(parent)
            elif name == "expert_output_weight":
                value[:2].copy_(2.0 * parent)
                value[2:].zero_()
            elif name == "router_output_weight":
                value[:, :2].copy_(parent)
                value[:, 2:].copy_(parent)
            elif name == "router_bias":
                value[:2].copy_(parent)
                value[2:].copy_(parent)
            else:
                value.copy_(parent)
    return target


def _synthetic_complete_payload(
) -> tuple[dict[str, object], dict[str, object]]:
    _require_sources()
    protocol, objective_protocol, c2_protocol, d3_recipe = (
        runner._authenticated_declarations()
    )
    problem = runner._prepare_live_fit_problem(
        source_expert_rank_path=runner.DEFAULT_SOURCE_EXPERT_RANK,
        source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
        source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
        basis_package_path=runner.DEFAULT_BASIS_PACKAGE,
        basis_package_file_sha256=(
            runner.DEFAULT_BASIS_PACKAGE_FILE_SHA256
        ),
        basis_package_payload_sha256=(
            runner.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        ),
        cache_dir=None,
        device_name="cpu",
        dtype="float32",
        _protocol_override=protocol,
    )
    sources = problem.sources
    expert_rank_result = sources.expert_rank_result
    control = sources.expert_rank_plan
    lifted = _lift_standard_e2_executor(
        control,
        target_config=runner._executor_config(protocol.e4_executor),
    )
    treatment = replace(
        control,
        executor_artifact=lifted.artifact_state_dict(),
        artifact_sha256="",
    )
    treatment = replace(
        treatment,
        final_metrics=(
            runner.evaluate_contrast_aware_reference_provider(
                treatment,
                batches=problem.fit_batches,
                contrast_pairs=problem.fit_pairs,
            )
        ),
        artifact_sha256="",
    )
    plans = {
        "fp_expert_count.primary.expert2": control,
        "fp_expert_count.primary.expert4": treatment,
    }
    initialization = runner._initial_audit_contract(
        protocol,
        pair_role="primary",
        control_initial_metrics_sha256=(
            control.initial_metrics.artifact_sha256
        ),
        treatment_initial_metrics_sha256=(
            treatment.initial_metrics.artifact_sha256
        ),
    )
    frozen_gradient = runner._preflight_binding_for_role(
        protocol,
        "primary",
    )["treatment_gradient"]
    assert isinstance(frozen_gradient, dict)
    gradients = {
        "expert2": runner._gradient_audit(
            applicable=False,
            pair_role="primary",
            protocol=protocol,
            values={
                name: 0.0
                for name in frozen_gradient
            },
        ),
        "expert4": runner._gradient_audit(
            applicable=True,
            pair_role="primary",
            protocol=protocol,
            values=frozen_gradient,
        ),
    }
    parities = {
        "expert2": _not_applicable_parity("primary"),
        "expert4": _treatment_parity(
            treatment,
            pair_role="primary",
        ),
    }
    ordinary_gates = runner._deferred_collision_gates(
        runner.SyntheticReferenceGates()
    )
    contrast_gates = runner.ContrastAssessmentGates()
    raw_teacher_energy = float(
        expert_rank_result.report["gauge"][
            "raw_fit_teacher_weighted_energy"
        ]
    )
    training_teacher_energy = float(
        expert_rank_result.report["gauge"][
            "unit_fit_teacher_weighted_energy"
        ]
    )
    training_signal = expert_rank_result.report[
        "training_teacher_signal_diagnostics"
    ]
    assert isinstance(training_signal, dict)
    controls = problem.controls
    rows: dict[str, dict[str, object]] = {}
    ordinary_candidates: dict[
        str,
        runner.FullWidthReferenceCandidate,
    ] = {}
    for candidate_id, plan in plans.items():
        arm = candidate_id.rsplit(".", 1)[-1]
        stored = 31_492 if arm == "expert2" else 48_166
        support_radius = runner.c2._feature_radius(
            plan,
            problem.fit,
        )
        (
            candidate,
            score,
            raw_predictions,
            structural_metadata,
        ) = runner.d0d3._fit_only_ordinary_candidate_and_score(
            candidate_id=candidate_id,
            plan=plan,
            measured=problem.fit,
            ordinary_probes=problem.ordinary_probes,
            controls=controls,
            metric_weight=problem.raw_metric_weight,
            standardized_gauge_sha256=(
                problem.standardized_gauge_sha256
            ),
            support_radius=support_radius,
            gates=problem.fidelity_gates,
        )
        ordinary_candidates[candidate_id] = candidate
        assert candidate.stored_scalar_count == stored
        ordinary_flags = runner.r64._recompute_ordinary_gate_flags(
            score,
            ordinary_gates,
        )
        (
            contrast,
            identities,
            coverage,
        ) = runner.d0d3._fit_contrast_assessment(
            protocol=c2_protocol,
            measured=problem.fit,
            predictions=raw_predictions,
            metric_weight=problem.raw_metric_weight,
            gates=contrast_gates,
            required_null_candidate_pass_count=24,
        )
        contrast_result = contrast.state_dict()
        balance = runner.d0d3._contribution_balance_gate(
            plan,
            recipe=d3_recipe,
            gates=objective_protocol.gates,
            training_teacher_energy=training_teacher_energy,
            raw_teacher_energy=raw_teacher_energy,
            teacher_signal_diagnostics=training_signal,
        )
        ordinary_pass = all(ordinary_flags.values())
        fit_pass = (
            ordinary_pass
            and contrast_result["overall_status"] == "pass"
            and bool(coverage["every_teacher_qualified_contrast_passed"])
            and bool(coverage["all_families_cover_all_four_rank_bands"])
            and bool(
                coverage["required_null_contrasts_valid_and_passed"]
            )
        )
        rows[candidate_id] = {
            "candidate_id": candidate_id,
            "pair_role": "primary",
            "arm": arm,
            "seed": protocol.training.primary_seed,
            "outer_rank": 64,
            "expert_count": plan.executor_config.expert_count,
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
            "gradient_openness": copy.deepcopy(gradients[arm]),
            "postfit_wrapper_concat_parity": copy.deepcopy(parities[arm]),
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
            "contrast_identities": identities,
            "contrast_coverage": coverage,
            "structural_metadata": structural_metadata,
            "accounting": asdict(plan.accounting()),
            "execution_accounting": (
                runner.d0d3._fit_execution_accounting(
                    plan,
                    problem.fit_batches,
                )
            ),
            "fit_capability_contract": {
                "ordinary_gate_count": len(ordinary_flags),
                "all_ordinary_gates_passed": ordinary_pass,
                "all_contrast_families_passed": (
                    contrast_result["overall_status"] == "pass"
                ),
                "every_qualified_contrast_passed": bool(
                    coverage["every_teacher_qualified_contrast_passed"]
                ),
                "all_four_rank_bands_covered": bool(
                    coverage[
                        "all_families_cover_all_four_rank_bands"
                    ]
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
        "gradient_open_expert4": True,
        "postfit_wrapper_concat_parity": True,
        "expert2_balance": bool(
            rows["fp_expert_count.primary.expert2"][
                "objective_balance_gate"
            ]["passed"]  # type: ignore[index]
        ),
        "expert4_balance": bool(
            rows["fp_expert_count.primary.expert4"][
                "objective_balance_gate"
            ]["passed"]  # type: ignore[index]
        ),
        "expert2_source_sequences": True,
        "expert4_source_sequences": True,
        "expert2_primary_replay": True,
        "fixed_outer_encoder_decoder_router": True,
        "exact_training_contract": True,
    }
    assert all(validity_flags.values())
    status = runner._pair_status(
        bool(
            rows["fp_expert_count.primary.expert2"][
                "fit_capability_pass"
            ]
        ),
        bool(
            rows["fp_expert_count.primary.expert4"][
                "fit_capability_pass"
            ]
        ),
    )
    for row in rows.values():
        row["pair_treatment_validity"] = {
            "passed": True,
            "flags": validity_flags,
            "failure_semantics": (
                "invalid_paired_expert_count_comparison_no_capacity_"
                "conclusion"
            ),
        }
        row["pair_comparison_status"] = status
    decision = runner._recompute_decision_from_rows(rows)
    executed = (
        "fp_expert_count.primary.expert2",
        "fp_expert_count.primary.expert4",
    )
    result_hashes = {
        candidate_id: runner._json_sha256(
            rows[candidate_id],
            domain=runner._RESULT_DOMAIN,
        )
        for candidate_id in executed
    }
    source_manifest = expert_rank_result.manifest
    source = protocol.source
    code = runner._code_sha256s()
    replay_batches = runner._ordered_ordinary_scoring_batches(
        problem.fit_batches
    )
    replay_pairs = tuple(
        sorted(problem.fit_pairs, key=lambda value: value.pair_id)
    )
    manifest = {
        "schema": runner._SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "source_expert_rank_protocol_sha256": source.expert_rank_protocol_sha256,
        "source_expert_rank_code_bundle_sha256": (
            source.expert_rank_code_bundle_sha256
        ),
        "source_expert_rank_logical_artifact_sha256": (
            source.expert_rank_logical_artifact_sha256
        ),
        "source_expert_rank_tensor_file_sha256": (
            source.expert_rank_tensor_file_sha256
        ),
        "source_expert_rank_report_sha256": source.expert_rank_report_sha256,
        "source_expert_rank_primary_e2r64_plan_sha256": (
            source.expert_rank_primary_e2r64_plan_sha256
        ),
        "source_expert_rank_primary_e2r64_result_sha256": (
            source.expert_rank_primary_e2r64_result_sha256
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
        "ordinary_candidate_sha256s": {
            candidate_id: ordinary_candidates[
                candidate_id
            ].artifact_sha256
            for candidate_id in executed
        },
        "ordinary_scoring_indexed_batch_sha256s": tuple(
            value.artifact_sha256 for value in replay_batches
        ),
        "fit_contrast_pair_sha256s": tuple(
            value.artifact_sha256 for value in replay_pairs
        ),
        "candidate_result_sha256s": result_hashes,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_expert_rank_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "fit_split_wrapper_used": True,
        "published_plans_use_concatenated_executor": True,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code,
        "code_bundle_sha256": runner._code_bundle_sha256(code),
        "scientific_scope": (
            "fit_only_paired_function_preserving_expert_count_control"
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
        "calibration_state": copy.deepcopy(
            expert_rank_result.state["calibration_state"]
        ),
        "unit_rms_gauge_state": copy.deepcopy(
            expert_rank_result.state["unit_rms_gauge_state"]
        ),
        "canonical_metric_weight": expert_rank_result.state[
            "canonical_metric_weight"
        ].clone(),
        "controls_state": copy.deepcopy(
            expert_rank_result.state["controls_state"]
        ),
        "ordinary_scoring_batch_states": tuple(
            value.state_dict() for value in replay_batches
        ),
        "fit_contrast_pair_states": tuple(
            value.state_dict() for value in replay_pairs
        ),
        "plan_states": {
            candidate_id: plans[candidate_id].state_dict()
            for candidate_id in executed
        },
        "ordinary_candidate_states": {
            candidate_id: ordinary_candidates[
                candidate_id
            ].state_dict()
            for candidate_id in executed
        },
        "candidate_results": rows,
    }
    report = {
        **manifest,
        "artifact_sha256": logical,
        "protocol": protocol.state_dict(),
        "calibration": copy.deepcopy(expert_rank_result.report["calibration"]),
        **{
            name: copy.deepcopy(expert_rank_result.report[name])
            for name in runner._MEASUREMENT_FIELDS
        },
        "candidate_results": [
            copy.deepcopy(rows[candidate_id])
            for candidate_id in executed
        ],
        "interpretation": runner._interpretation_from_evidence(
            rows,
            decision,
        ),
        "safety": runner._safety_contract(),
    }
    return state, report


@pytest.fixture(scope="module")
def complete_payload() -> tuple[dict[str, object], dict[str, object]]:
    return _synthetic_complete_payload()


def _payload_copy(
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    return copy.deepcopy(complete_payload)


def _refresh_outer_bindings(
    state: dict[str, object],
    report: dict[str, object],
    *,
    recompute_decision: bool = True,
) -> None:
    manifest = state["manifest"]
    rows = state["candidate_results"]
    assert isinstance(manifest, dict)
    assert isinstance(rows, dict)
    executed = tuple(manifest["executed_candidate_ids"])
    if recompute_decision:
        decision = runner._recompute_decision_from_rows(rows)
        manifest.update(decision)
    else:
        decision = {
            name: manifest[name] for name in runner._DECISION_FIELDS
        }
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
    report["interpretation"] = runner._interpretation_from_evidence(
        rows,
        decision,
    )


def _publish_and_load(
    state: dict[str, object],
    report: dict[str, object],
    *,
    output: Path,
) -> object:
    receipt = runner._publish_artifact(state, report, output=output)
    return runner.load_function_preserving_expert_count_control_artifact(
        output,
        **receipt,
    )


def _refresh_published_tensor_receipt(
    output: Path,
    receipt: dict[str, str],
) -> None:
    tensor_payload = output.read_bytes()
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    report_path = output.with_suffix(".json")
    published_report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = published_report["artifact"]
    assert isinstance(artifact, dict)
    artifact["tensor_file_sha256"] = tensor_sha256
    artifact["tensor_file_bytes"] = len(tensor_payload)
    published_report.pop("report_sha256")
    report_sha256 = runner._json_sha256(
        published_report,
        domain=runner._REPORT_DOMAIN,
    )
    published_report["report_sha256"] = report_sha256
    report_path.write_text(
        json.dumps(
            published_report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt.update(
        {
            "expected_tensor_file_sha256": tensor_sha256,
            "expected_report_sha256": report_sha256,
        }
    )


def _replace_ordinary_candidate(
    state: dict[str, object],
    *,
    candidate_id: str,
    candidate: runner.FullWidthReferenceCandidate,
) -> None:
    candidates = state["ordinary_candidate_states"]
    rows = state["candidate_results"]
    manifest = state["manifest"]
    assert isinstance(candidates, dict)
    assert isinstance(rows, dict)
    assert isinstance(manifest, dict)
    row = rows[candidate_id]
    assert isinstance(row, dict)
    score = runner.FullWidthCandidateScore.from_state_dict(
        row["ordinary_score"]
    )
    rebound_score = replace(
        score,
        candidate_artifact_sha256=candidate.artifact_sha256,
        structural_metrics=candidate.structural_metrics,
        artifact_sha256="",
    )
    candidates[candidate_id] = candidate.state_dict()
    row["candidate_binding_sha256"] = candidate.artifact_sha256
    row["ordinary_score"] = rebound_score.state_dict()
    ordinary_hashes = manifest["ordinary_candidate_sha256s"]
    assert isinstance(ordinary_hashes, dict)
    ordinary_hashes[candidate_id] = candidate.artifact_sha256


def test_complete_synthetic_artifact_publishes_and_strictly_reloads(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    output = (tmp_path / "valid-expert-count.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)
    tensor_payload = output.read_bytes()
    consumed = torch.load(
        output,
        map_location="cpu",
        weights_only=True,
    )

    assert isinstance(consumed, dict)
    assert runner._canonical_torch_payload(consumed) == tensor_payload
    loaded = runner.load_function_preserving_expert_count_control_artifact(
        output,
        **receipt,
    )

    assert loaded.manifest["outcome"] == "primary_both_fail"
    assert loaded.manifest["e8_expert_count_control_authorized"] is True
    assert loaded.manifest["fresh_c3_authorized"] is False


def test_public_loader_requires_triple_before_deserialization(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, report = _payload_copy(complete_payload)
    output = (tmp_path / "missing-receipt.pt").resolve()
    runner._publish_artifact(state, report, output=output)
    calls = {"torch_load": 0}

    def bomb(*_args: object, **_kwargs: object) -> object:
        calls["torch_load"] += 1
        raise AssertionError("deserialization must not be reached")

    monkeypatch.setattr(runner.torch, "load", bomb)
    with pytest.raises(TypeError):
        runner.load_function_preserving_expert_count_control_artifact(
            output
        )  # type: ignore[call-arg]
    assert calls["torch_load"] == 0


def test_tensor_trust_anchor_is_checked_before_deserialization(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, report = _payload_copy(complete_payload)
    output = (tmp_path / "wrong-tensor-anchor.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)
    calls = {"torch_load": 0}

    def bomb(*_args: object, **_kwargs: object) -> object:
        calls["torch_load"] += 1
        raise AssertionError("deserialization must not be reached")

    monkeypatch.setattr(runner.torch, "load", bomb)
    receipt["expected_tensor_file_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="external trust anchor mismatch: tensor file",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )
    assert calls["torch_load"] == 0


def test_loader_rejects_trailing_bytes_with_refreshed_receipt(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    output = (tmp_path / "trailing-payload.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)
    appended = b"verbatim private prompt"
    with output.open("ab") as handle:
        handle.write(appended)
    _refresh_published_tensor_receipt(output, receipt)

    with pytest.raises(ValueError, match="contains trailing bytes"):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_loader_rejects_ignored_zip_member_with_refreshed_receipt(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    output = (tmp_path / "extra-member.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)
    forged = (tmp_path / "extra-member-forged.pt").resolve()
    reader = torch._C.PyTorchFileReader(str(output))
    writer = torch._C.PyTorchFileWriter(str(forged))
    for name in reader.get_all_records():
        payload = reader.get_record(name)
        writer.write_record(name, payload, len(payload))
    secret = b"verbatim private prompt"
    writer.write_record("verbatim_private_prompt", secret, len(secret))
    writer.write_end_of_file()
    forged.replace(output)
    tensor_payload = output.read_bytes()

    runner._require_canonical_torch_zip_framing(tensor_payload)
    consumed = torch.load(
        output,
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(consumed, dict)
    assert consumed["artifact_sha256"] == state["artifact_sha256"]
    assert runner._canonical_torch_payload(consumed) != tensor_payload
    _refresh_published_tensor_receipt(output, receipt)

    with pytest.raises(
        ValueError,
        match="not the canonical serialization of its consumed state",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_publisher_rejects_structural_prompt_key(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_expert_count.primary.expert2"]
    assert isinstance(row, dict)
    structural = row["structural_metadata"]
    assert isinstance(structural, dict)
    structural["prompt"] = "verbatim private prompt"
    output = (tmp_path / "forbidden-structural-prompt.pt").resolve()

    with pytest.raises(
        ValueError,
        match="structural_metadata.prompt is forbidden",
    ):
        runner._publish_artifact(state, report, output=output)
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_original_receipt_rejects_self_rehashed_outcome(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    valid_state, valid_report = _payload_copy(complete_payload)
    forged_state, forged_report = _payload_copy(complete_payload)
    output = (tmp_path / "anchored-outcome.pt").resolve()
    trusted = runner._publish_artifact(
        valid_state,
        valid_report,
        output=output,
    )
    output.unlink()
    output.with_suffix(".json").unlink()
    forged_manifest = forged_state["manifest"]
    assert isinstance(forged_manifest, dict)
    forged_manifest["outcome"] = "primary_both_pass"
    _refresh_outer_bindings(
        forged_state,
        forged_report,
        recompute_decision=False,
    )
    runner._publish_artifact(
        forged_state,
        forged_report,
        output=output,
    )

    with pytest.raises(ValueError, match="external trust anchor mismatch"):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **trusted,
        )


def test_run_validates_two_step_preflight_before_any_full_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = (
        runner.default_function_preserving_expert_count_control_protocol()
    )
    problem = SimpleNamespace(
        protocol=protocol,
        measurement_evidence={},
    )
    events: list[str] = []

    monkeypatch.setattr(
        runner,
        "_prepare_live_fit_problem",
        lambda **_kwargs: problem,
    )
    monkeypatch.setattr(
        runner,
        "_measurement_evidence_sha256",
        lambda _value: protocol.training.measurement_evidence_sha256,
    )

    def preflight(_problem: object) -> dict[str, object]:
        events.append("two_step_preflight")
        return {"sealed": True}

    def validate(
        _preflight: object,
        *,
        protocol: object,
    ) -> None:
        assert protocol is problem.protocol
        events.append("preflight_validation")

    def reject_full_fit(**_kwargs: object) -> object:
        events.append("full_fit")
        raise RuntimeError("stop before 600-step fit")

    monkeypatch.setattr(runner, "_run_fit_only_preflight", preflight)
    monkeypatch.setattr(runner, "_validate_fit_only_preflight", validate)
    monkeypatch.setattr(runner, "_evaluate_pair", reject_full_fit)

    with pytest.raises(RuntimeError, match="stop before 600-step fit"):
        runner.run_function_preserving_expert_count_control(
            output=(tmp_path / "preflight-order.pt").resolve()
        )
    assert events == [
        "two_step_preflight",
        "preflight_validation",
        "full_fit",
    ]


@pytest.fixture(scope="module")
def live_two_step_problem() -> object:
    _require_sources()
    protocol, *_ = runner._authenticated_declarations()
    return runner._prepare_live_fit_problem(
        source_expert_rank_path=runner.DEFAULT_SOURCE_EXPERT_RANK,
        source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
        source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
        basis_package_path=runner.DEFAULT_BASIS_PACKAGE,
        basis_package_file_sha256=(
            runner.DEFAULT_BASIS_PACKAGE_FILE_SHA256
        ),
        basis_package_payload_sha256=(
            runner.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        ),
        cache_dir=None,
        device_name="cpu",
        dtype="float32",
        _protocol_override=protocol,
    )


@pytest.mark.parametrize("pair_role", ["primary", "replication"])
def test_expert2_two_step_delegates_exactly_to_expert_rank_treatment(
    pair_role: str,
    live_two_step_problem: object,
) -> None:
    problem = live_two_step_problem
    protocol = problem.protocol  # type: ignore[attr-defined]
    seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    data = runner.contrast_fit._prepare_fit_data(
        fit_batches=problem.fit_batches,  # type: ignore[attr-defined]
        contrast_pairs=problem.fit_pairs,  # type: ignore[attr-defined]
        require_fit_split=True,
    )
    control = runner._new_control_model(
        modal_center=problem.modal_center,  # type: ignore[attr-defined]
        gain_log_center=problem.gain_log_center,  # type: ignore[attr-defined]
        gain_log_scale=problem.gain_log_scale,  # type: ignore[attr-defined]
        residual_width=problem.basis.residual_width,  # type: ignore[attr-defined]
        rms_epsilon=problem.epsilon,  # type: ignore[attr-defined]
        target_center=problem.target_center,  # type: ignore[attr-defined]
        target_scale=problem.target_scale,  # type: ignore[attr-defined]
        seed=seed,
    )
    plan, count_gradient, count_parity = (
        runner._fit_authenticated_expert_rank_control(
            control,
            data=data,
            target_center=problem.target_center,  # type: ignore[attr-defined]
            target_scale=problem.target_scale,  # type: ignore[attr-defined]
            metric_weight=problem.unit_gauge.metric_weight,  # type: ignore[attr-defined]
            objective=runner._objective(protocol),
            steps=2,
            learning_rate=protocol.training.learning_rate,
            seed=seed,
            pair_role=pair_role,
            synthetic_binding_sha256=(  # type: ignore[attr-defined]
                problem.fit_data_binding_sha256
            ),
            protocol=protocol,
        )
    )
    source_protocol = (
        runner.default_function_preserving_expert_rank_control_protocol()
    )
    source_frozen = source_protocol.preflight.for_role(pair_role)
    source_two_step = source_frozen["two_step_postfit_parity"]
    count_two_step = protocol.preflight.for_role(pair_role)[
        "control_two_step"
    ]
    assert isinstance(source_two_step, dict)
    assert isinstance(count_two_step, dict)
    expected_metrics_sha256 = source_two_step["metrics_sha256"]
    expected_executor_sha256 = source_two_step[
        "concatenated_executor_sha256"
    ]

    assert count_two_step == {
        "metrics_sha256": expected_metrics_sha256,
        "executor_sha256": expected_executor_sha256,
    }
    assert plan.final_metrics.artifact_sha256 == expected_metrics_sha256
    assert (
        runner.contrast_fit._executor_artifact_sha256(
            plan.executor_artifact
        )
        == expected_executor_sha256
    )

    count_gradient_fields = {
        name: value
        for name, value in count_gradient.items()
        if name.startswith("step") and name.endswith("_norm")
    }
    assert count_gradient["pair_role"] == pair_role
    assert count_gradient["artifact_kind"] == (
        "fisher_graph.expert_count_gradient_openness"
    )
    assert count_gradient["format_version"] == runner._FORMAT_VERSION
    assert count_gradient["applicable"] is False
    assert count_gradient["passed"] is True
    assert count_gradient_fields
    assert set(count_gradient_fields.values()) == {0.0}
    assert all(count_gradient["flags"].values())
    unhashed_gradient = dict(count_gradient)
    gradient_sha256 = unhashed_gradient.pop("artifact_sha256")
    assert gradient_sha256 == runner._json_sha256(
        unhashed_gradient,
        domain=runner._AUDIT_DOMAIN,
    )

    expected_count_parity = {
        "artifact_kind": (
            "fisher_graph.expert_count_wrapper_concat_parity"
        ),
        "format_version": runner._FORMAT_VERSION,
        "pair_role": pair_role,
        "stage": "not_applicable_control",
        "applicable": False,
        "passed": True,
    }
    expected_count_parity["artifact_sha256"] = runner._json_sha256(
        expected_count_parity,
        domain=runner._AUDIT_DOMAIN,
    )
    assert count_parity == expected_count_parity


def test_primary_source_plan_guard_precedes_expert4_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = (
        runner.default_function_preserving_expert_count_control_protocol()
    )
    frozen = protocol.preflight.for_role("primary")
    problem = SimpleNamespace(
        protocol=protocol,
        fit_batches=(),
        fit_pairs=(),
        modal_center=object(),
        gain_log_center=0.0,
        gain_log_scale=1.0,
        basis=SimpleNamespace(residual_width=1),
        epsilon=1e-6,
        target_center=object(),
        target_scale=object(),
        unit_gauge=SimpleNamespace(metric_weight=object()),
        fit_data_binding_sha256="0" * 64,
    )
    control = object()
    treatment = object()
    e4_fit_called = False

    monkeypatch.setattr(
        runner.contrast_fit,
        "_prepare_fit_data",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner,
        "_new_control_model",
        lambda **_kwargs: control,
    )
    monkeypatch.setattr(
        runner,
        "_new_treatment_model",
        lambda **_kwargs: treatment,
    )
    monkeypatch.setattr(
        runner,
        "_initial_equivalence",
        lambda *_args, **_kwargs: {
            "passed": True,
            "artifact_sha256": frozen["initialization_audit_sha256"],
        },
    )
    forged_control_plan = SimpleNamespace(
        artifact_sha256="f" * 64,
        initial_metrics=SimpleNamespace(
            artifact_sha256=(
                protocol.source
                .expert_rank_primary_e2r64_initial_metrics_sha256
            )
        ),
        final_metrics=SimpleNamespace(
            artifact_sha256=(
                protocol.source
                .expert_rank_primary_e2r64_final_metrics_sha256
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "_fit_authenticated_expert_rank_control",
        lambda *_args, **_kwargs: (
            forged_control_plan,
            {"passed": True},
            {"passed": True},
        ),
    )

    def forbidden_e4_fit(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        nonlocal e4_fit_called
        e4_fit_called = True
        raise AssertionError("E4 fit ran before the E2 source-plan guard")

    monkeypatch.setattr(
        runner,
        "_fit_from_initialized_model",
        forbidden_e4_fit,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "primary E2 did not exactly replay the authenticated "
            "expert-rank treatment plan"
        ),
    ):
        runner._evaluate_pair(
            pair_role="primary",
            seed=protocol.training.primary_seed,
            problem=problem,
        )
    assert e4_fit_called is False


def test_dormant_child_lift_is_function_preserving_and_gradient_open(
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, _ = complete_payload
    plan_state = state["plan_states"][
        "fp_expert_count.primary.expert2"
    ]
    plan = ContrastAwareReferenceProviderPlan.from_state_dict(plan_state)
    protocol = runner.default_function_preserving_expert_count_control_protocol()
    base = ResidualGatedCausalModalExecutor.from_artifact_state_dict(
        plan.executor_artifact
    )
    lifted = _lift_standard_e2_executor(
        plan,
        target_config=runner._executor_config(protocol.e4_executor),
    )
    coordinates = torch.randn(2, 7, 66, dtype=base.dtype)
    mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    positions = torch.arange(7, dtype=torch.int64).expand(2, -1)

    base_output = base(
        coordinates,
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
    )
    lifted_output = lifted(
        coordinates,
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
    )
    assert torch.allclose(
        base_output,
        lifted_output,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(
        lifted.expert_input_weight[:2],
        lifted.expert_input_weight[2:],
    )
    assert torch.count_nonzero(lifted.expert_output_weight[2:]) == 0
    frozen = protocol.preflight.for_role("primary")["treatment_gradient"]
    assert isinstance(frozen, dict)
    assert frozen["step1_dormant_base_input_gradient_norm"] == 0.0
    assert frozen["step1_dormant_base_output_gradient_norm"] > 1e-12
    assert frozen["step1_router_sibling_gradient_norm"] > 1e-12
    assert frozen["step2_dormant_base_input_gradient_norm"] > 1e-12


@pytest.mark.parametrize(
    "source_flag",
    [
        "replication_executed",
        "two_seed_inner_expert_rank_supported",
        "descending_expert_rank_ladder_authorized",
        "fresh_c3_authorized",
    ],
)
def test_source_authorization_rejects_expert_rank_outcome_drift(
    source_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_sources()
    protocol, *_ = runner._authenticated_declarations()
    source = protocol.source
    loaded = (
        runner.expert_rank.load_function_preserving_expert_rank_control_artifact(
        runner.DEFAULT_SOURCE_EXPERT_RANK,
        expected_artifact_sha256=source.expert_rank_logical_artifact_sha256,
        expected_tensor_file_sha256=source.expert_rank_tensor_file_sha256,
        expected_report_sha256=source.expert_rank_report_sha256,
        )
    )
    manifest = dict(loaded.manifest)
    manifest[source_flag] = True
    tampered = replace(loaded, manifest=manifest)
    monkeypatch.setattr(
        runner.expert_rank,
        "load_function_preserving_expert_rank_control_artifact",
        lambda *_args, **_kwargs: tampered,
    )

    with pytest.raises(
        ValueError,
        match="authenticated expert-rank source identity drifted",
    ):
        runner._authenticate_sources(
            source_expert_rank_path=runner.DEFAULT_SOURCE_EXPERT_RANK,
            source_diagnostic_path=runner.DEFAULT_SOURCE_DIAGNOSTIC,
            source_rank64_path=runner.DEFAULT_SOURCE_RANK64,
            protocol=protocol,
        )


def _decision_rows(
    *,
    primary: tuple[bool, bool],
    replication: tuple[bool, bool] | None = None,
) -> dict[str, object]:
    rows: dict[str, object] = {}

    def add_pair(role: str, values: tuple[bool, bool]) -> None:
        initial = {"passed": True}
        gradient = {"passed": True}
        parity = {"passed": True}
        balance = {"passed": True}
        sequence = {"passed": True}
        flags = {
            "initial_observable_and_jvp_equivalence": True,
            "gradient_open_expert4": True,
            "postfit_wrapper_concat_parity": True,
            "expert2_balance": True,
            "expert4_balance": True,
            "expert2_source_sequences": True,
            "expert4_source_sequences": True,
            "expert2_primary_replay": True,
            "fixed_outer_encoder_decoder_router": True,
            "exact_training_contract": True,
        }
        validity = {
            "passed": True,
            "flags": flags,
            "failure_semantics": (
                "invalid_paired_expert_count_comparison_no_capacity_"
                "conclusion"
            ),
        }
        status = runner._pair_status(*values)
        for index, arm in enumerate(("expert2", "expert4")):
            rows[f"fp_expert_count.{role}.{arm}"] = {
                "initialization_equivalence": initial,
                "gradient_openness": gradient,
                "postfit_wrapper_concat_parity": parity,
                "objective_balance_gate": balance,
                "source_sequence_comparison": sequence,
                "source_replay_exact": role == "primary" and index == 0,
                "fit_capability_pass": values[index],
                "pair_treatment_validity": validity,
                "pair_comparison_status": status,
            }

    add_pair("primary", primary)
    if replication is not None:
        add_pair("replication", replication)
    return rows


@pytest.mark.parametrize(
    ("primary", "replication", "outcome", "ladder", "e8"),
    [
        ((False, False), None, "primary_both_fail", False, True),
        (
            (False, True),
            (False, True),
            "two_seed_routed_expert_count_support",
            True,
            False,
        ),
        (
            (False, True),
            (False, False),
            "inconsistent_replication_both_fail",
            False,
            False,
        ),
        ((True, True), None, "primary_both_pass", False, False),
    ],
)
def test_decision_matrix(
    primary: tuple[bool, bool],
    replication: tuple[bool, bool] | None,
    outcome: str,
    ladder: bool,
    e8: bool,
) -> None:
    decision = runner._recompute_decision_from_rows(
        _decision_rows(primary=primary, replication=replication)
    )

    assert decision["outcome"] == outcome
    assert decision["descending_expert_count_ladder_authorized"] is ladder
    assert decision["e8_expert_count_control_authorized"] is e8
    assert decision["fresh_c3_authorized"] is False
    assert decision["compression_claim_authorized"] is False
    assert decision["speed_claim_authorized"] is False


def test_treatment_pass_without_replication_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="authorized expert-count replication is absent",
    ):
        runner._recompute_decision_from_rows(
            _decision_rows(primary=(False, True))
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        (
            "execution_accounting",
            "semantics",
            "verbatim private prompt",
            "execution accounting replay drifted",
        ),
        (
            "execution_accounting",
            "canonical_total_mac_count",
            True,
            "execution accounting replay drifted",
        ),
            (
                "gradient_openness",
                "step2_dormant_base_input_gradient_norm",
                0.0,
                "gradient binding drifted",
        ),
        (
            "postfit_wrapper_concat_parity",
            "concatenated_metrics_sha256",
            "verbatim private prompt",
            "postfit parity binding drifted",
        ),
    ],
)
def test_strict_loader_rejects_self_rehashed_nested_tamper(
    target: str,
    field: str,
    value: object,
    match: str,
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    treatment = rows["fp_expert_count.primary.expert4"]
    assert isinstance(treatment, dict)
    nested = treatment[target]
    assert isinstance(nested, dict)
    nested[field] = value
    if target in {
        "gradient_openness",
        "postfit_wrapper_concat_parity",
    }:
        unhashed = dict(nested)
        unhashed.pop("artifact_sha256")
        nested["artifact_sha256"] = runner._json_sha256(
            unhashed,
            domain=runner._AUDIT_DOMAIN,
        )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / f"tamper-{target}-{field}.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match=match):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_safety_tamper(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    safety = report["safety"]
    assert isinstance(safety, dict)
    safety["contains_prompt_text"] = True
    output = (tmp_path / "tamper-safety.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="report safety semantics drifted"):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_score_probe_transplant(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_expert_count.primary.expert4"]
    assert isinstance(row, dict)
    score = runner.FullWidthCandidateScore.from_state_dict(
        row["ordinary_score"]
    )
    probe_metrics = list(score.probe_metrics)
    probe_metrics[0] = replace(
        probe_metrics[0],
        probe_id=(
            "development_c2.fit.ordinary.block_sparse."
            "00_verbatim_private_prompt"
        ),
    )
    forged = replace(
        score,
        probe_metrics=tuple(probe_metrics),
        artifact_sha256="",
    )
    row["ordinary_score"] = forged.state_dict()
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-score-probe.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="ordinary score replay drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_prediction_plus_123(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    candidate_id = "fp_expert_count.primary.expert4"
    candidates = state["ordinary_candidate_states"]
    assert isinstance(candidates, dict)
    candidate = runner._restore_ordinary_candidate(
        candidates[candidate_id]
    )
    predictions = list(candidate.predictions)
    predictions[0] = replace(
        predictions[0],
        retained_standardized_prediction=(
            predictions[0].retained_standardized_prediction + 123.0
        ),
        artifact_sha256="",
    )
    forged = replace(
        candidate,
        predictions=tuple(predictions),
        artifact_sha256="",
    )
    _replace_ordinary_candidate(
        state,
        candidate_id=candidate_id,
        candidate=forged,
    )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-prediction-plus-123.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="ordinary candidate numerical replay drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_structural_value(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_expert_count.primary.expert2"]
    assert isinstance(row, dict)
    structural = row["structural_metadata"]
    assert isinstance(structural, dict)
    structural["support_radius"] = float(structural["support_radius"]) + 1.0
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-structural-value.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="ordinary structural replay drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_execution_accounting(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_expert_count.primary.expert2"]
    assert isinstance(row, dict)
    execution = row["execution_accounting"]
    assert isinstance(execution, dict)
    execution["fit_panel_core_mac_count"] += 1
    execution["fit_panel_total_mac_count"] += 1
    execution["macs_per_valid_row_over_fit_panel"] = (
        execution["fit_panel_total_mac_count"]
        / execution["fit_panel_valid_rows"]
    )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-execution-accounting.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="execution accounting replay drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_contrast_coverage(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    rows = state["candidate_results"]
    assert isinstance(rows, dict)
    row = rows["fp_expert_count.primary.expert2"]
    assert isinstance(row, dict)
    coverage = row["contrast_coverage"]
    assert isinstance(coverage, dict)
    coverage["required_null_contrasts_valid_and_passed"] = False
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-contrast-coverage.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(ValueError, match="contrast replay drifted"):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_fit_target_sequence(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    batch_states = state["ordinary_scoring_batch_states"]
    manifest = state["manifest"]
    assert isinstance(batch_states, tuple)
    assert isinstance(manifest, dict)
    batches = [
        runner.IndexedReferenceBatch.from_state_dict(value)
        for value in batch_states
    ]
    first = batches[0]
    changed_batch = replace(
        first.batch,
        target_modes=first.batch.target_modes + 1.0,
        content_sha256="",
        artifact_sha256="",
    )
    batches[0] = replace(
        first,
        batch=changed_batch,
        artifact_sha256="",
    )
    batches.sort(key=lambda value: value.artifact_sha256)
    state["ordinary_scoring_batch_states"] = tuple(
        value.state_dict() for value in batches
    )
    manifest["ordinary_scoring_indexed_batch_sha256s"] = tuple(
        value.artifact_sha256 for value in batches
    )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-fit-target-sequence.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="fit replay protocol commitment drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_fit_pair_sequence(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    pair_states = state["fit_contrast_pair_states"]
    manifest = state["manifest"]
    assert isinstance(pair_states, tuple)
    assert isinstance(manifest, dict)
    pairs = [
        runner.ReferenceProviderContrastPair.from_state_dict(value)
        for value in pair_states
    ]
    index = next(
        position
        for position, value in enumerate(pairs)
        if value.teacher_midpoint_jvp is not None
    )
    pair = pairs[index]
    assert pair.teacher_midpoint_jvp is not None
    pairs[index] = replace(
        pair,
        teacher_midpoint_jvp=pair.teacher_midpoint_jvp + 1.0,
        artifact_sha256="",
    )
    state["fit_contrast_pair_states"] = tuple(
        value.state_dict() for value in pairs
    )
    manifest["fit_contrast_pair_sha256s"] = tuple(
        value.artifact_sha256 for value in pairs
    )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-fit-pair-sequence.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="fit replay protocol commitment drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_strict_loader_rejects_self_rehashed_final_metrics(
    tmp_path: Path,
    complete_payload: tuple[dict[str, object], dict[str, object]],
) -> None:
    state, report = _payload_copy(complete_payload)
    candidate_id = "fp_expert_count.primary.expert4"
    plans = state["plan_states"]
    rows = state["candidate_results"]
    manifest = state["manifest"]
    candidates = state["ordinary_candidate_states"]
    assert isinstance(plans, dict)
    assert isinstance(rows, dict)
    assert isinstance(manifest, dict)
    assert isinstance(candidates, dict)
    plan = ContrastAwareReferenceProviderPlan.from_state_dict(
        plans[candidate_id]
    )
    delta = 1e-6
    forged_metrics = replace(
        plan.final_metrics,
        pointwise_mse=plan.final_metrics.pointwise_mse + delta,
        weighted_total=(
            plan.final_metrics.weighted_total
            + plan.objective.pointwise_weight * delta
        ),
        artifact_sha256="",
    )
    forged_plan = replace(
        plan,
        final_metrics=forged_metrics,
        artifact_sha256="",
    )
    plans[candidate_id] = forged_plan.state_dict()
    plan_hashes = manifest["candidate_plan_sha256s"]
    assert isinstance(plan_hashes, dict)
    plan_hashes[candidate_id] = forged_plan.artifact_sha256

    candidate = runner._restore_ordinary_candidate(
        candidates[candidate_id]
    )
    rebound_candidate = replace(
        candidate,
        candidate_binding_sha256=forged_plan.artifact_sha256,
        artifact_sha256="",
    )
    _replace_ordinary_candidate(
        state,
        candidate_id=candidate_id,
        candidate=rebound_candidate,
    )
    row = rows[candidate_id]
    assert isinstance(row, dict)
    row["plan_sha256"] = forged_plan.artifact_sha256
    row["final_training_metrics"] = forged_metrics.state_dict()
    row["final_contribution_audit"] = (
        runner.audit_objective_contributions(
            forged_metrics,
            forged_plan.objective,
        ).state_dict()
    )
    parity = row["postfit_wrapper_concat_parity"]
    assert isinstance(parity, dict)
    parity["concatenated_metrics_sha256"] = (
        forged_metrics.artifact_sha256
    )
    parity_unhashed = dict(parity)
    parity_unhashed.pop("artifact_sha256")
    parity["artifact_sha256"] = runner._json_sha256(
        parity_unhashed,
        domain=runner._AUDIT_DOMAIN,
    )
    _refresh_outer_bindings(state, report)
    output = (tmp_path / "tamper-final-metrics.pt").resolve()
    receipt = runner._publish_artifact(state, report, output=output)

    with pytest.raises(
        ValueError,
        match="final metrics replay drifted",
    ):
        runner.load_function_preserving_expert_count_control_artifact(
            output,
            **receipt,
        )


def test_safety_distinguishes_loaded_regenerated_and_hashed_state() -> None:
    safety = runner._safety_contract()

    assert (
        safety[
            "contains_loaded_expert_rank_source_final_provider_parameters"
        ]
        is False
    )
    assert (
        safety[
            "contains_regenerated_expert_rank_final_equivalent_provider_"
            "parameters"
        ]
        is True
    )
    assert (
        safety[
            "contains_regenerated_expert_rank_initial_parameter_hashes"
        ]
        is True
    )
    assert safety["contains_expert2_provider_parameters"] is True
    assert safety["contains_expert4_provider_parameters"] is True
    assert safety["contains_synthetic_fit_teacher_target_tensors"] is True
    assert safety["contains_raw_fit_targets"] is True
    assert safety["contains_teacher_jvp_tensors"] is True
    assert safety["contains_provider_chart_jvp_tensors"] is True
    assert safety["contains_raw_model_teacher_hidden_states"] is False
    assert "contains_regenerated_width_initial_parameters" not in safety
