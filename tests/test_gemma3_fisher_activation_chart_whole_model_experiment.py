from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_fisher_activation_chart_whole_model_experiment as experiment,
)
from fisher_graph.gemma3_fisher_activation_chart_whole_model_experiment_setup import (
    build_fisher_activation_chart_whole_model_experiment_setup,
    write_fisher_activation_chart_whole_model_experiment_setup,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Payload:
    def __init__(self, name: str, **values: object) -> None:
        self.name = name
        self.values = values


def _splits() -> dict[str, object]:
    return {
        "A_fit": _Payload("A_fit", row_ids=("fit.0", "fit.1")),
        "A_select": _Payload("A_select", row_ids=("select.0",)),
        "B_guard": _Payload("B_guard", row_ids=("guard.0",)),
        "C_assessment": _Payload("C_assessment", row_ids=("assess.0",)),
    }


_FIT_ARMS = (
    "progressive_independent_factorized_chart",
    "progressive_factorized_chart_transport",
    "transport_row_shuffled_negative_control",
)
_EVALUATED_ARMS = (
    *_FIT_ARMS[:2],
    "transport_zero_edge_control",
    _FIT_ARMS[2],
)
_EXECUTED_ARMS = ("native", *_EVALUATED_ARMS)


def _zero_native_projection_counts() -> dict[str, dict[str, int]]:
    return {
        str(layer): {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
        for layer in range(18)
    }


def _delegated_resource(*, edge_count: int) -> dict[str, object]:
    source_parameters = experiment.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    native_mlp_parameters = 18 * 3 * 640 * 2048
    per_layer_compiled_parameters = 2_776_616
    compiled_parameters = (
        18 * per_layer_compiled_parameters + edge_count * 4 * 4
    )
    candidate_parameters = (
        source_parameters - native_mlp_parameters + compiled_parameters
    )
    return {
        "native_target_mlp_parameter_scalars": native_mlp_parameters,
        "logical_compiled_coefficient_scalars": compiled_parameters,
        "logical_candidate_whole_model_scalars": candidate_parameters,
        "logical_whole_model_reduction_fraction": (
            (source_parameters - candidate_parameters) / source_parameters
        ),
        "estimated_compiled_macs_per_token": compiled_parameters,
        "routing_softmax_logits_per_token": 144,
        "sparse_edge_index_bytes": 32 * edge_count,
        "theoretical_sparse_cost_reported": True,
        "realized_kernel_latency_measured": False,
        "shadow_runtime_retains_native_source_weights": True,
        "source_free_serialized_candidate_materialized": False,
        "realized_resident_memory_reduction_claimed": False,
        "compression_success_claimed": False,
    }


def _synthetic_delegated_transport_report() -> dict[str, object]:
    source_parameters = experiment.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    edge_counts = {
        "progressive_independent_factorized_chart": 0,
        "progressive_factorized_chart_transport": 16,
        "transport_zero_edge_control": 16,
        "transport_row_shuffled_negative_control": 16,
    }
    protocol = experiment.build_cross_layer_chart_transport_protocol()
    report: dict[str, object] = {
        "schema": experiment.CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA,
        "format_version": (
            experiment.CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION
        ),
        "scientific_status": {
            "development_only": True,
            "partial_protocol_predictive_screen": True,
            "full_preregistered_ladder_executed": False,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "all_modes_retained": True,
            "deletion_or_merging_performed": False,
            "compression_success_claimed": False,
            "dependency_proven_by_prediction_alone": False,
        },
        "execution_scope": {
            "executed_arms": list(_EXECUTED_ARMS),
            "unexecuted_protocol_arms": [
                "exact_all_mode_graph",
                "one_shot_independent_dense_chart_reproduction",
                "budget_matched_factorized_chart_transport",
            ],
            "deferred_required_metrics": [
                "prefix_delta_nll_and_kl_after_each_layer",
                "worst_family_metrics_and_bootstrap_intervals",
                "held_direction_jvp_and_finite_displacement",
            ],
            "compression_criterion_evaluated": False,
            "causal_confirmation_evaluated": False,
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "model": {
            "model_id": experiment.PINNED_MODEL_ID,
            "revision": experiment.DEFAULT_REVISION,
            "model_fingerprint": _digest("synthetic-model"),
            "device": "cpu",
            "dtype": "float32",
            "source_whole_model_parameters": source_parameters,
        },
        "data": {
            "panel_file": "synthetic-panel.json",
            "panel_sha256": experiment.PINNED_DEVELOPMENT_PANEL_SHA256,
            "family_count": 6,
            "fit_example_count": 30,
            "evaluation_example_count": 30,
            "fit_example_ids_sha256": _digest("fit-example-ids"),
            "evaluation_example_ids_sha256": _digest("eval-example-ids"),
            "prompt_disjoint": True,
            "family_disjoint": False,
        },
        "configuration": {
            "generator_rank": 4,
            "chart_temperature": 1.0,
            "affine_relative_ridge": 1.0e-3,
            "local_chart_relative_ridge": 1.0e-2,
            "maximum_incoming_charts_per_target": 2,
            "transport_ridge": 1.0e-4,
            "row_shuffled_control_executed": True,
            "same_token_adjacent_layer_only": True,
        },
        "native": {"nll_per_token": 1.0, "supervised_tokens": 10},
        "progressive_layer_fit": {
            arm: {str(layer): {} for layer in range(18)}
            for arm in _FIT_ARMS
        },
        "progressive_freeze_receipts": {
            arm: [{"layer_ordinal": layer} for layer in range(18)]
            for arm in _FIT_ARMS
        },
        "full_model_evaluation": {
            arm: {"nll_per_token": 1.1} for arm in _EVALUATED_ARMS
        },
        "full_model_comparisons": {},
        "execution_receipts": {
            arm: {
                "compiled_layer_calls_per_batch": 18,
                "native_source_projection_call_counts": (
                    _zero_native_projection_counts()
                ),
                "transport_edge_count": edge_counts[arm],
            }
            for arm in _EVALUATED_ARMS
        },
        "resources": {
            arm: _delegated_resource(edge_count=edge_counts[arm])
            for arm in _FIT_ARMS
        },
        "timing": {
            "progressive_fit_seconds": 0.0,
            "full_model_evaluation_seconds": 0.0,
        },
    }
    report["report_sha256"] = experiment._transport_report_sha256(report)
    return report


def _synthetic_dependencies(
    calls: list[str],
) -> experiment.FisherActivationChartWorkerDependencies:
    def capture(setup: object, fit: object):
        calls.append("capture_teacher_fisher_rows")
        capture_contract = setup["teacher_capture_contract"]
        assert capture_contract["source_role"] == "A_fit"
        assert capture_contract["full_Fisher_coupling_matrix_computed"] is False
        assert capture_contract["outer_product_matrix_computed"] is False
        assert isinstance(fit, _Payload) and fit.name == "A_fit"
        payload = _Payload(
            "teacher_capture",
            activation_rows=((1.0, 0.0), (0.0, 2.0)),
            score_gradient_rows=((0.5, -0.5), (0.25, 0.75)),
            fisher_row_weights=(0.25, 2.25),
        )
        return experiment.FisherActivationChartStageArtifact(
            phase="capture_teacher_fisher_rows",
            payload=payload,
            metadata={
                "source_role": "A_fit",
                "objective": "summed_next_token_negative_log_likelihood",
                "fisher_estimator": (
                    "scalar_row_weight_from_squared_modal_state_times_"
                    "summed_NLL_score_gradient"
                ),
                "full_fisher_matrix_computed": False,
                "outer_product_matrix_computed": False,
                "valid_row_count": 2,
                "activation_rows_sha256": _digest("activation-rows"),
                "score_gradient_rows_sha256": _digest("score-gradient-rows"),
                "fisher_row_weights_sha256": _digest("fisher-row-weights"),
                "row_identity_sha256": _digest("fit-row-identity"),
                "raw_rows_serialized": False,
            },
        )

    def fit_charts(setup: object, capture_stage: object):
        calls.append("fit_chart_generators")
        assert capture_stage.phase == "capture_teacher_fisher_rows"
        assert capture_stage.payload.name == "teacher_capture"
        assert "activation_rows" in capture_stage.payload.values
        assert "fisher_row_weights" in capture_stage.payload.values
        assert capture_stage.metadata["fisher_row_weights_sha256"] == _digest(
            "fisher-row-weights"
        )
        payload = _Payload(
            "charts",
            source_capture=capture_stage.payload,
            chart_router="hidden_state_chart_coordinates",
        )
        return experiment.FisherActivationChartStageArtifact(
            phase="fit_chart_generators",
            payload=payload,
            metadata={
                "input_capture_sha256": capture_stage.artifact_sha256,
                "chart_count": setup["frozen_hyperparameters"]["chart_count"],
                "generator_family_ids": ["family.alpha", "family.beta"],
                "generator_invocation_ids": [
                    f"mutation.{index}" for index in range(2, 37, 2)
                ],
                "family_identity_is_parameter_identity": True,
                "invocation_identity_is_temporal_identity": True,
                "clustering_feature_names": [
                    "Fisher_coupling",
                    "activation_chart_overlap",
                    "teacher_displacement_alignment",
                    "held_direction_JVP_similarity",
                ],
                "provenance_fields": [
                    "source_layer_ordinal",
                    "source_stage",
                ],
                "layer_provenance_used_as_clustering_feature": False,
                "cross_provenance_discovery_performed": True,
            },
        )

    def compile_first(setup: object, charts: object):
        del setup
        calls.append("compile_first_candidate")
        assert charts.phase == "fit_chart_generators"
        return experiment.FisherActivationChartStageArtifact(
            phase="compile_first_candidate",
            payload=_Payload("first_candidate", charts=charts.payload),
            metadata={
                "chart_generator_sha256": charts.artifact_sha256,
                "runtime_event_count": 39,
                "replaced_feed_forward_mutations": 18,
                "transport_fit_applied": False,
            },
        )

    def recapture(setup: object, first: object, fit: object):
        del setup
        calls.append("recapture_candidate_trajectory")
        assert first.phase == "compile_first_candidate"
        assert first.payload.name == "first_candidate"
        assert isinstance(fit, _Payload) and fit.name == "A_fit"
        return experiment.FisherActivationChartStageArtifact(
            phase="recapture_candidate_trajectory",
            payload=_Payload(
                "candidate_trajectory",
                candidate=first.payload,
                candidate_chart_coordinates=((0.1, 0.2), (0.4, -0.1)),
            ),
            metadata={
                "first_candidate_sha256": first.artifact_sha256,
                "source_role": "A_fit",
                "candidate_prefix_trajectory_used": True,
                "teacher_states_substituted_for_candidate_states": False,
                "trajectory_rows_sha256": _digest("candidate-trajectory"),
            },
        )

    def fit_transports(setup: object, charts: object, trajectory: object):
        calls.append("fit_candidate_transports")
        assert charts.phase == "fit_chart_generators"
        assert trajectory.phase == "recapture_candidate_trajectory"
        assert trajectory.payload.name == "candidate_trajectory"
        assert trajectory.metadata["candidate_prefix_trajectory_used"] is True
        return experiment.FisherActivationChartStageArtifact(
            phase="fit_candidate_transports",
            payload=_Payload(
                "transports",
                candidate_chart_coordinates=trajectory.payload.values[
                    "candidate_chart_coordinates"
                ],
            ),
            metadata={
                "chart_generator_sha256": charts.artifact_sha256,
                "candidate_trajectory_sha256": trajectory.artifact_sha256,
                "transport_rank": setup["frozen_hyperparameters"][
                    "transport_rank"
                ],
                "fit_on_candidate_trajectory": True,
                "teacher_rows_used_as_transport_source_states": False,
                "causal_predecessor_only": True,
            },
        )

    def compile_frozen(setup: object, charts: object, transports: object):
        del setup
        calls.append("compile_frozen_candidate")
        assert charts.phase == "fit_chart_generators"
        assert transports.phase == "fit_candidate_transports"
        return experiment.FisherActivationChartStageArtifact(
            phase="compile_frozen_candidate",
            payload=_Payload(
                "frozen_candidate",
                charts=charts.payload,
                transports=transports.payload,
            ),
            metadata={
                "chart_generator_sha256": charts.artifact_sha256,
                "transport_sha256": transports.artifact_sha256,
                "runtime_event_count": 39,
                "tied_lexical_physical_storage_count": 1,
                "all_18_feed_forward_mutations_replaced": True,
                "frozen_before_evaluation": True,
            },
        )

    def evaluate(setup: object, candidate: object, evaluation_splits: object):
        calls.append("evaluate_controls")
        assert candidate.phase == "compile_frozen_candidate"
        assert list(evaluation_splits) == ["A_select", "B_guard", "C_assessment"]
        assert all(
            isinstance(split, _Payload) and split.name == role
            for role, split in evaluation_splits.items()
        )
        assert "A_fit" not in evaluation_splits
        required = setup["actual_evaluation_conditions"]
        source_parameters = experiment.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        candidate_parameters = source_parameters - 18
        compiled_parameters = 20
        resident_parameters = source_parameters + compiled_parameters
        all_fitted_stack_parameters = 3 * compiled_parameters
        worker_resident_parameters = (
            source_parameters + all_fitted_stack_parameters
        )
        return experiment.FisherActivationChartStageArtifact(
            phase="evaluate_controls",
            payload=_Payload("evaluation", candidate=candidate.payload),
            metadata={
                "evaluated_conditions": required,
                "negative_controls_present": [
                    "transport_zero_edge_control",
                    "transport_row_shuffled_negative_control",
                ],
                "fit_rows_seen": False,
                "later_role_refit_count": 0,
                "resources": {
                    "source_unique_parameters": source_parameters,
                    "candidate_unique_parameters": candidate_parameters,
                    "compiled_graph_coefficient_scalars": compiled_parameters,
                    "physical_resident_overlay_scalars": resident_parameters,
                    "all_fitted_stack_coefficient_scalars": (
                        all_fitted_stack_parameters
                    ),
                    "development_worker_resident_lower_bound_scalars": (
                        worker_resident_parameters
                    ),
                    "removed_parameters": 18,
                    "parameter_reduction_fraction": (
                        18 / source_parameters
                    ),
                    "inventory_closes": True,
                    "copied_exact_coefficients_counted": True,
                    "tied_lexical_parameters_counted_once": True,
                    "physical_resident_overlay_inventory_closes": True,
                    "physical_resident_scope": (
                        "selected_arm_only_not_development_worker_peak"
                    ),
                    "projected_logical_candidate_float32_bytes": (
                        4 * candidate_parameters
                    ),
                    "physical_resident_overlay_float32_bytes": (
                        4 * resident_parameters
                    ),
                    "development_worker_resident_lower_bound_float32_bytes": (
                        4 * worker_resident_parameters
                    ),
                    "serialized_candidate_artifact_written": False,
                    "whole_model_MAC_inventory_closes": False,
                    "resident_memory_reduction_claimed": False,
                },
                "logical_macs_by_component": {
                    "exact_attention": 1000,
                    "chart_generators": 200,
                    "transports": 25,
                },
                "copied_exact_coefficients_counted": True,
                "source_calls_by_condition": {
                    condition: 1 for condition in required
                },
                "overlay_runtime": True,
            },
        )

    def qualify(setup: object, evaluation: object):
        del setup
        calls.append("qualify")
        assert evaluation.phase == "evaluate_controls"
        resources = evaluation.metadata["resources"]
        assert resources["removed_parameters"] == (
            resources["source_unique_parameters"]
            - resources["candidate_unique_parameters"]
        )
        assert resources["parameter_reduction_fraction"] == (
            18 / experiment.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        )
        return experiment.FisherActivationChartStageArtifact(
            phase="qualify",
            payload=None,
            metadata={
                "evaluation_sha256": evaluation.artifact_sha256,
                "resource_inventory_closes": True,
                "projected_byte_inventory_closes": True,
                "required_negative_controls_present": True,
                "family_disjoint_guard_passed": False,
                "serialized_candidate_artifact_present": False,
                "source_free_runtime_established": False,
                "compression_success_claimed": False,
                "compression_claim_authorized": False,
                "latency_claimed": False,
                "qualified": False,
            },
        )

    return experiment.FisherActivationChartWorkerDependencies(
        capture_teacher_fisher_rows=capture,
        fit_chart_generators=fit_charts,
        compile_first_candidate=compile_first,
        recapture_candidate_trajectory=recapture,
        fit_candidate_transports=fit_transports,
        compile_frozen_candidate=compile_frozen,
        evaluate_controls=evaluate,
        qualify=qualify,
    )


def test_synthetic_orchestration_routes_fisher_rows_through_chart_runtime() -> None:
    calls: list[str] = []
    report = experiment._orchestrate_fisher_activation_chart_worker(
        setup=build_fisher_activation_chart_whole_model_experiment_setup(),
        splits=_splits(),
        dependencies=_synthetic_dependencies(calls),
    )

    assert calls == [
        "capture_teacher_fisher_rows",
        "fit_chart_generators",
        "compile_first_candidate",
        "recapture_candidate_trajectory",
        "fit_candidate_transports",
        "compile_frozen_candidate",
        "evaluate_controls",
        "qualify",
    ]
    assert [receipt["phase"] for receipt in report["phase_receipts"]] == calls
    assert report["phase_receipts"][1]["metadata"][
        "input_capture_sha256"
    ] == report["phase_receipts"][0]["artifact_sha256"]
    assert report["phase_receipts"][4]["metadata"][
        "candidate_trajectory_sha256"
    ] == report["phase_receipts"][3]["artifact_sha256"]
    assert report["phase_receipts"][5]["metadata"][
        "transport_sha256"
    ] == report["phase_receipts"][4]["artifact_sha256"]
    assert report["phase_receipts"][6]["metadata"][
        "negative_controls_present"
    ] == [
        "transport_zero_edge_control",
        "transport_row_shuffled_negative_control",
    ]
    assert report["qualification"]["resource_inventory_closes"] is True
    assert report["qualification"]["qualified"] is False
    assert report["scientific_status"] == {
        "bounded_partial_arm": True,
        "future_protocol_test_seam_only": True,
        "default_model_facing_worker_executed": False,
        "unrestricted_primary_executed": False,
        "compression_success_claimed": False,
        "source_free_runtime_claimed": False,
        "latency_claimed": False,
    }
    raw = json.dumps(report, sort_keys=True)
    assert '"activation_rows":' not in raw
    assert '"score_gradient_rows":' not in raw
    assert '"fisher_row_weights":' not in raw
    assert '"candidate_chart_coordinates":' not in raw
    assert "_Payload" not in raw


def test_stage_receipts_authenticate_the_frozen_dependency_dag() -> None:
    report = experiment._orchestrate_fisher_activation_chart_worker(
        setup=build_fisher_activation_chart_whole_model_experiment_setup(),
        splits=_splits(),
        dependencies=_synthetic_dependencies([]),
    )
    receipts = report["phase_receipts"]
    assert receipts[0]["input_artifact_sha256s"] == []
    assert receipts[1]["input_artifact_sha256s"] == [
        receipts[0]["artifact_sha256"]
    ]
    assert receipts[2]["input_artifact_sha256s"] == [
        receipts[1]["artifact_sha256"]
    ]
    assert receipts[3]["input_artifact_sha256s"] == [
        receipts[2]["artifact_sha256"]
    ]
    assert receipts[4]["input_artifact_sha256s"] == [
        receipts[1]["artifact_sha256"],
        receipts[3]["artifact_sha256"],
    ]
    assert receipts[5]["input_artifact_sha256s"] == [
        receipts[1]["artifact_sha256"],
        receipts[4]["artifact_sha256"],
    ]
    assert receipts[6]["input_artifact_sha256s"] == [
        receipts[5]["artifact_sha256"]
    ]
    assert receipts[7]["input_artifact_sha256s"] == [
        receipts[6]["artifact_sha256"]
    ]
    assert all(receipt["payload_serialized_in_report"] is False for receipt in receipts)


def test_orchestrator_fails_closed_on_split_or_stage_drift() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    dependencies = _synthetic_dependencies([])
    missing = _splits()
    del missing["B_guard"]
    with pytest.raises(ValueError, match="four frozen roles"):
        experiment._orchestrate_fisher_activation_chart_worker(
            setup=setup,
            splits=missing,
            dependencies=dependencies,
        )

    def wrong_capture(_setup: object, _fit: object):
        return experiment.FisherActivationChartStageArtifact(
            phase="fit_chart_generators",
            payload=None,
            metadata={},
        )

    wrong = deepcopy(dependencies)
    object.__setattr__(wrong, "capture_teacher_fisher_rows", wrong_capture)
    with pytest.raises(ValueError, match="wrong phase"):
        experiment._orchestrate_fisher_activation_chart_worker(
            setup=setup,
            splits=_splits(),
            dependencies=wrong,
        )


def test_stage_metadata_must_be_finite_and_json_safe() -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        experiment.FisherActivationChartStageArtifact(
            phase="qualify",
            payload=None,
            metadata={"bad": float("nan")},
        )
    with pytest.raises(ValueError, match="phase is not frozen"):
        experiment.FisherActivationChartStageArtifact(
            phase="train_on_guard",
            payload=None,
            metadata={},
        )


def test_model_facing_runner_is_not_invoked_by_synthetic_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "run_gemma3_cross_layer_chart_transport_model_experiment",
        lambda **_kwargs: pytest.fail("real model-facing experiment was invoked"),
    )
    report = experiment._orchestrate_fisher_activation_chart_worker(
        setup=build_fisher_activation_chart_whole_model_experiment_setup(),
        splits=_splits(),
        dependencies=_synthetic_dependencies([]),
    )
    assert report["execution_kind"] == (
        "future_full_protocol_injectable_orchestration_seam"
    )
    assert report["report_sha256"] == experiment._sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def test_synthetic_report_strict_round_trip(tmp_path: Path) -> None:
    report = experiment._orchestrate_fisher_activation_chart_worker(
        setup=build_fisher_activation_chart_whole_model_experiment_setup(),
        splits=_splits(),
        dependencies=_synthetic_dependencies([]),
    )
    destination = tmp_path / "synthetic-chart-worker.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert (
        experiment.load_fisher_activation_chart_whole_model_experiment_report(
            destination
        )
        == report
    )


@pytest.mark.parametrize(
    "tamper",
    (
        lambda report: report["phase_receipts"][1]["metadata"].__setitem__(
            "chart_count", 999
        ),
        lambda report: report["resource_accounting"].__setitem__(
            "removed_parameters", 17
        ),
        lambda report: report["qualification"].__setitem__(
            "compression_success_claimed", True
        ),
    ),
)
def test_rehashed_synthetic_report_tampering_is_rejected(tamper) -> None:
    report = deepcopy(
        experiment._orchestrate_fisher_activation_chart_worker(
            setup=build_fisher_activation_chart_whole_model_experiment_setup(),
            splits=_splits(),
            dependencies=_synthetic_dependencies([]),
        )
    )
    tamper(report)
    report["report_sha256"] = experiment._sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    with pytest.raises(ValueError):
        experiment.validate_fisher_activation_chart_whole_model_experiment_report(
            report
        )


def test_model_facing_runner_delegates_only_the_frozen_development_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_path = tmp_path / "setup.json"
    output = tmp_path / "worker-report.json"
    write_fisher_activation_chart_whole_model_experiment_setup(setup_path)
    delegated_calls: list[dict[str, object]] = []

    def delegated(**kwargs: object) -> dict[str, object]:
        delegated_calls.append(dict(kwargs))
        return _synthetic_delegated_transport_report()

    monkeypatch.setattr(
        experiment,
        "run_gemma3_cross_layer_chart_transport_model_experiment",
        delegated,
    )
    report = experiment.run_gemma3_fisher_activation_chart_whole_model_experiment(
        setup_path=setup_path,
        transport_protocol_path=tmp_path / "unopened-transport-protocol.json",
        panel_path=tmp_path / "unopened-panel.json",
        output=output,
    )

    assert len(delegated_calls) == 1
    delegated_call = delegated_calls[0]
    assert delegated_call["generator_rank"] == 4
    assert delegated_call["maximum_incoming_charts_per_target"] == 2
    assert delegated_call["transport_ridge"] == 1.0e-4
    assert delegated_call["include_row_shuffled_control"] is True
    assert delegated_call["batch_size"] == 4
    assert "A_select" not in delegated_call
    assert "B_guard" not in delegated_call
    assert "C_assessment" not in delegated_call

    assert report == (
        experiment.load_fisher_activation_chart_whole_model_experiment_report(
            output
        )
    )
    assert report["bounded_transport_report"]["configuration"] == {
        "generator_rank": 4,
        "chart_temperature": 1.0,
        "affine_relative_ridge": 1.0e-3,
        "local_chart_relative_ridge": 1.0e-2,
        "maximum_incoming_charts_per_target": 2,
        "transport_ridge": 1.0e-4,
        "row_shuffled_control_executed": True,
        "same_token_adjacent_layer_only": True,
    }
    assert report["bounded_transport_report"]["execution_scope"][
        "executed_arms"
    ] == list(_EXECUTED_ARMS)
    assert report["scientific_status"]["family_disjoint_evaluation"] is False
    assert report["qualification"]["qualified"] is False
    assert report["qualification"]["compression_success_claimed"] is False
    assert report["runtime_scope"]["batch_size"] == 4
    resources = report["resource_accounting"]
    assert resources["tied_lexical_parameters_counted_once"] is True
    assert resources["inventory_closes"] is True
    assert resources["physical_resident_overlay_inventory_closes"] is True
    assert resources["physical_resident_overlay_scalars"] > resources[
        "source_unique_parameters"
    ]
    assert resources["all_fitted_stack_coefficient_scalars"] > resources[
        "compiled_graph_coefficient_scalars"
    ]
    assert resources["development_worker_resident_lower_bound_scalars"] == (
        resources["source_unique_parameters"]
        + resources["all_fitted_stack_coefficient_scalars"]
    )
    assert resources["physical_resident_scope"] == (
        "selected_arm_only_not_development_worker_peak"
    )
    assert resources["whole_model_MAC_inventory_closes"] is False
    assert resources["resident_memory_reduction_claimed"] is False


@pytest.mark.parametrize(
    "overrides",
    (
        {"generator_rank": 8},
        {"maximum_incoming_charts_per_target": 3},
        {"batch_size": 5},
        {"transport_ridge": 1.0e-3},
        {"include_row_shuffled_control": False},
    ),
)
def test_runner_rejects_frozen_argument_drift_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    setup_path = tmp_path / "setup.json"
    write_fisher_activation_chart_whole_model_experiment_setup(setup_path)
    monkeypatch.setattr(
        experiment,
        "run_gemma3_cross_layer_chart_transport_model_experiment",
        lambda **_kwargs: pytest.fail("delegated worker must not run"),
    )
    with pytest.raises(ValueError, match="frozen setup"):
        experiment.run_gemma3_fisher_activation_chart_whole_model_experiment(
            setup_path=setup_path,
            output=tmp_path / "report.json",
            **overrides,
        )


def _rehash_nested_and_outer(
    report: dict[str, object],
    *,
    rehash_nested: bool,
) -> None:
    nested = report["bounded_transport_report"]
    if rehash_nested:
        nested["report_sha256"] = experiment._transport_report_sha256(
            {
                key: value
                for key, value in nested.items()
                if key != "report_sha256"
            }
        )
    report["report_sha256"] = experiment._sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


@pytest.mark.parametrize(
    ("tamper", "rehash_nested"),
    (
        (
            lambda nested: nested.__setitem__("schema", "tampered.schema"),
            True,
        ),
        (
            lambda nested: nested.__setitem__("report_sha256", "0" * 64),
            False,
        ),
        (
            lambda nested: nested["configuration"].__setitem__(
                "maximum_incoming_charts_per_target", 3
            ),
            True,
        ),
        (
            lambda nested: nested["progressive_layer_fit"][
                "progressive_factorized_chart_transport"
            ].pop("17"),
            True,
        ),
        (
            lambda nested: nested["execution_scope"]["executed_arms"].pop(),
            True,
        ),
        (
            lambda nested: nested["resources"][
                "progressive_factorized_chart_transport"
            ].__setitem__(
                "logical_candidate_whole_model_scalars",
                nested["resources"][
                    "progressive_factorized_chart_transport"
                ]["logical_candidate_whole_model_scalars"]
                + 1,
            ),
            True,
        ),
    ),
)
def test_rehashed_nested_transport_tampering_is_rejected(
    tamper,
    rehash_nested: bool,
) -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    report = experiment._wrap_bounded_transport_report(
        setup=setup,
        transport_report=_synthetic_delegated_transport_report(),
    )
    tampered = deepcopy(report)
    tamper(tampered["bounded_transport_report"])
    _rehash_nested_and_outer(tampered, rehash_nested=rehash_nested)
    with pytest.raises(ValueError):
        experiment.validate_fisher_activation_chart_whole_model_experiment_report(
            tampered
        )


def test_coordinated_compiled_resource_tamper_is_structurally_rejected() -> None:
    report = experiment._wrap_bounded_transport_report(
        setup=build_fisher_activation_chart_whole_model_experiment_setup(),
        transport_report=_synthetic_delegated_transport_report(),
    )
    tampered = deepcopy(report)
    source = tampered["resource_accounting"]["source_unique_parameters"]
    nested_resource = tampered["bounded_transport_report"]["resources"][
        "progressive_factorized_chart_transport"
    ]
    delta = 16
    nested_resource["logical_compiled_coefficient_scalars"] += delta
    nested_resource["logical_candidate_whole_model_scalars"] += delta
    nested_resource["logical_whole_model_reduction_fraction"] = (
        (
            source
            - nested_resource["logical_candidate_whole_model_scalars"]
        )
        / source
    )

    resources = tampered["resource_accounting"]
    resources["compiled_graph_coefficient_scalars"] += delta
    resources["candidate_unique_parameters"] += delta
    resources["physical_resident_overlay_scalars"] += delta
    resources["all_fitted_stack_coefficient_scalars"] += delta
    resources["development_worker_resident_lower_bound_scalars"] += delta
    resources["removed_parameters"] = (
        source - resources["candidate_unique_parameters"]
    )
    resources["parameter_reduction_fraction"] = (
        resources["removed_parameters"] / source
    )
    resources["projected_logical_candidate_float32_bytes"] = (
        4 * resources["candidate_unique_parameters"]
    )
    resources["physical_resident_overlay_float32_bytes"] = (
        4 * resources["physical_resident_overlay_scalars"]
    )
    resources["development_worker_resident_lower_bound_float32_bytes"] = (
        4 * resources["development_worker_resident_lower_bound_scalars"]
    )
    _rehash_nested_and_outer(tampered, rehash_nested=True)

    with pytest.raises(ValueError, match="selected resource ledger"):
        experiment.validate_fisher_activation_chart_whole_model_experiment_report(
            tampered
        )
