from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fisher_graph import (
    gemma3_fisher_activation_chart_whole_model_experiment_setup as setup_contract,
)
from fisher_graph.gemma3_fisher_activation_chart_whole_model_experiment_setup import (
    DEFAULT_OUTPUT,
    FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID,
    FISHER_ACTIVATION_CHART_SETUP_FORMAT_VERSION,
    FISHER_ACTIVATION_CHART_SETUP_SCHEMA,
    FisherActivationChartWholeModelExperimentSetup,
    build_fisher_activation_chart_whole_model_experiment_setup,
    load_fisher_activation_chart_whole_model_experiment_setup,
    main,
    validate_fisher_activation_chart_whole_model_experiment_setup,
    write_fisher_activation_chart_whole_model_experiment_setup,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / (
    "src/fisher_graph/"
    "gemma3_fisher_activation_chart_whole_model_experiment_setup.py"
)


def test_setup_freezes_a_bounded_runnable_arm_without_running_it() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    assert setup["schema"] == FISHER_ACTIVATION_CHART_SETUP_SCHEMA
    assert setup["format_version"] == FISHER_ACTIVATION_CHART_SETUP_FORMAT_VERSION
    assert setup["artifact_role"] == (
        "setup_only_for_runnable_bounded_development_overlay"
    )
    assert setup["protocol_binding"]["implemented_bounded_arm"] == (
        FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID
    )
    assert setup["protocol_binding"][
        "delegated_executable_transport_protocol"
    ]["implemented_by_this_worker"] is True
    assert setup["protocol_binding"]["aspirational_layer_agnostic_protocol"][
        "implemented_by_this_worker"
    ] is False
    assert setup["setup_receipt"] == {
        "model_weights_loaded": False,
        "prompt_data_opened": False,
        "tokenization_performed": False,
        "activation_capture_performed": False,
        "fisher_computed": False,
        "charts_fitted": False,
        "candidate_trajectory_recaptured": False,
        "transports_fitted": False,
        "experiment_executed": False,
        "metrics_present": False,
        "compression_success_claimed": False,
    }
    scope = setup["bounded_scope"]
    assert scope["development_overlay_only"] is True
    assert scope["all_18_feed_forward_mutations_replaced"] is True
    assert scope["replacement_kind"] == (
        "progressive_fisher_weighted_factorized_activation_chart_"
        "generators_with_adjacent_same_token_transports"
    )
    assert scope["uses_native_tied_lexical_ingress_and_egress"] is True
    assert scope["uses_native_attention_residual_and_normalization_path"] is True
    assert scope["compiled_exact_39_event_control_executed"] is False
    assert scope["model_forward_overlay_used_for_evaluation"] is True
    assert scope["standalone_source_free_runtime_established"] is False
    assert scope["deployed_candidate_artifact_materialized"] is False
    assert setup["frozen_hyperparameters"] == {
        "chart_count": 8,
        "chart_rank": 16,
        "generator_rank": 4,
        "transport_rank": 4,
        "maximum_incoming_charts_per_target": 2,
        "batch_size": 4,
        "deterministic_seed": 1729,
        "affine_relative_ridge": 1.0e-3,
        "local_chart_relative_ridge": 1.0e-2,
        "transport_relative_ridge": 1.0e-4,
        "chart_temperature": 1.0,
    }
    assert json.loads(json.dumps(setup)) == setup


def test_phase_order_and_data_firewall_are_frozen() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    phases = setup["default_worker_phase_contract"]
    assert [phase["phase"] for phase in phases] == [
        "authenticate_setup_protocol_and_panel",
        "load_local_model_and_tokenize_A_fit_A_eval",
        "fit_progressive_independent_chart_stack",
        "fit_progressive_adjacent_transport_stack",
        "fit_row_shuffled_transport_negative_control",
        "evaluate_frozen_development_controls",
        "write_and_strictly_reload_wrapper_report",
    ]
    assert phases[2]["consumes"] == ["A_fit", "native_MLP_targets"]
    assert phases[3]["consumes"] == [
        "A_fit",
        "candidate_prefix_trajectories",
    ]
    assert phases[5]["consumes"] == ["A_eval", "three_frozen_stacks"]
    firewall = setup["data_firewall"]
    assert firewall["ordered_roles"] == ["A_fit", "A_eval"]
    assert firewall["prompt_and_example_id_disjoint_required"] is True
    assert firewall["family_disjoint"] is False
    assert firewall["heldout_confirmation_established"] is False
    assert firewall["A_eval_may_not_refit_any_tensor"] is True
    assert firewall["A_eval_rows_may_not_enter_capture_or_fit"] is True
    assert firewall["A_select_B_guard_C_assessment_used_by_default_worker"] is False
    future = setup["future_full_protocol_orchestration_seam"]
    assert future["implemented_by_default_worker"] is False
    assert future["roles"] == ["A_fit", "A_select", "B_guard", "C_assessment"]


def test_runtime_graph_has_exactly_39_events_and_18_chart_generators() -> None:
    graph = build_fisher_activation_chart_whole_model_experiment_setup()[
        "reference_transformer_schedule"
    ]
    events = graph["events"]
    assert len(events) == graph["event_count"] == 39
    assert graph["event_ids"] == [
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, 37)),
        "terminal.rmsnorm",
        "lexical.egress",
    ]
    assert [event["event_index"] for event in events] == list(range(39))
    mutations = events[1:37]
    assert sum(
        event["kind"] == "exact_causal_attention" for event in mutations
    ) == 18
    assert sum(
        event["kind"] == "fisher_activation_chart_generator"
        for event in mutations
    ) == 18
    assert all(
        event["kind"] == "exact_causal_attention"
        for event in mutations[0::2]
    )
    assert all(
        event["kind"] == "fisher_activation_chart_generator"
        for event in mutations[1::2]
    )
    assert graph["causal_order_is_strict"] is True
    assert graph["compiled_as_independent_39_event_executor"] is False
    assert graph["executed_through_native_model_forward_with_MLP_overlays"] is True


def test_capture_contract_feeds_fisher_weighted_activation_charts() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    capture = setup["teacher_capture_contract"]
    assert capture["source_role"] == "A_fit"
    assert capture["objective"] == "summed_next_token_negative_log_likelihood"
    assert capture["fisher_estimator"] == (
        "row_weight_from_squared_modal_state_times_summed_NLL_score_gradient"
    )
    assert capture["full_Fisher_coupling_matrix_computed"] is False
    assert capture["outer_product_matrix_computed"] is False
    assert capture["supervised_next_token_rows_only"] is False
    assert capture["final_valid_token_zero_gradient_row_is_retained"] is True
    assert set(capture["capture_fields"]) == {
        "normalized_feed_forward_inputs",
        "native_down_projection_modal_states",
        "score_gradients_wrt_modal_states",
        "example_id_and_logical_position_row_keys",
        "candidate_prefix_emitted_generator_latents",
    }
    assert capture["all_18_MLP_targets_captured_progressively"] is True
    assert capture["row_fisher_normalization"] == "mean_one_clamped_0p1_to_10"
    assert capture["raw_rows_remain_ephemeral"] is True
    assert capture["row_identity_sha256_required"] is True
    chart = setup["chart_and_generator_contract"]
    assert chart["atlas_input"] == (
        "Fisher_weighted_normalized_feed_forward_input_rows"
    )
    assert chart["routing_input"] == "current_normalized_feed_forward_input"
    assert chart["chart_center_and_tangent_basis_required"] is True


def test_generator_family_identity_is_separate_from_temporal_invocation() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    contract = setup["chart_and_generator_contract"]
    assert contract["generator_parameters_are_layer_specific"] is True
    assert contract["layer_ordinal_is_runtime_invocation_identity"] is True
    assert contract["cross_provenance_generator_families_discovered"] is False
    assert contract["cross_provenance_parameter_sharing_enabled"] is False

    for event in setup["reference_transformer_schedule"]["events"][1:37]:
        assert "source_layer_ordinal" not in event
        assert "source_stage" not in event
        assert set(event["provenance_only"]) == {
            "source_layer_ordinal",
            "source_stage",
            "source_mutation_id",
        }


def test_tied_lexical_storage_and_candidate_trajectory_refit_are_explicit() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    graph = setup["reference_transformer_schedule"]
    assert graph["native_tied_lexical_boundary_preserved"] is True
    assert graph["tied_lexical_physical_storage_counted_once"] is True
    assert setup["bounded_scope"][
        "uses_native_tied_lexical_ingress_and_egress"
    ] is True
    transport = setup["candidate_trajectory_transport_contract"]
    assert transport["progressive_prefix_frozen_before_next_layer_capture"] is True
    assert transport["candidate_A_fit_prefix_trajectory_recaptured"] is True
    assert transport[
        "teacher_rows_may_not_substitute_for_candidate_source_states"
    ] is True
    assert transport["fit_input"] == (
        "candidate_prefix_emitted_generator_latents"
    )
    assert transport["fit_output"] == (
        "target_chart_modal_residual_correction"
    )
    assert transport["causal_predecessor_only"] is True
    assert transport["current_executor_edge_scope"] == (
        "same_token_adjacent_layer_only"
    )


def test_negative_controls_and_resource_claim_firewall_are_required() -> None:
    setup = build_fisher_activation_chart_whole_model_experiment_setup()
    assert setup["actual_evaluation_conditions"] == [
        "native",
        "progressive_independent_factorized_chart",
        "progressive_factorized_chart_transport",
        "transport_zero_edge_control",
        "transport_row_shuffled_negative_control",
    ]
    deferred = set(setup["deferred_from_bounded_arm"])
    assert {
        "exact_39_event_control",
        "full_rank_chart_identity_control",
        "activation_only_no_Fisher_control",
        "matched_weight_SVD_body_control",
        "matched_deletion_control",
        "family_shuffled_control",
        "family_disjoint_guard_and_assessment",
    } <= deferred
    resources = setup["resource_and_claim_firewall"]
    assert resources["report_projected_unique_coefficient_scalars"] is True
    assert resources["report_projected_float32_coefficient_bytes"] is True
    assert resources["copied_exact_coefficients_must_be_counted"] is True
    assert resources["tied_lexical_storage_must_be_counted_once"] is True
    assert resources["overlay_runtime_is_not_source_free"] is True
    assert resources["projected_accounting_is_not_resident_memory_measurement"] is True
    assert resources["serialized_source_free_candidate_artifact_written"] is False
    assert resources["setup_artifact_authorizes_no_success_claim"] is True
    assert resources[
        "bounded_mechanics_rung_compression_success_authorized"
    ] is False
    assert resources["latency_claim_without_measurement"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["data_firewall"].__setitem__(
            "A_eval_may_not_refit_any_tensor", False
        ),
        lambda value: value["reference_transformer_schedule"].__setitem__(
            "event_count", 38
        ),
        lambda value: value["chart_and_generator_contract"].__setitem__(
            "cross_provenance_generator_families_discovered", True
        ),
        lambda value: value["candidate_trajectory_transport_contract"].__setitem__(
            "teacher_rows_may_not_substitute_for_candidate_source_states", False
        ),
        lambda value: value["bounded_scope"].__setitem__(
            "uses_native_tied_lexical_ingress_and_egress", False
        ),
        lambda value: value["frozen_hyperparameters"].__setitem__(
            "maximum_incoming_charts_per_target", 3
        ),
        lambda value: value["frozen_hyperparameters"].__setitem__(
            "batch_size", 5
        ),
        lambda value: value["frozen_hyperparameters"].__setitem__(
            "local_chart_relative_ridge", 1.0e-3
        ),
        lambda value: value["frozen_hyperparameters"].__setitem__(
            "chart_temperature", 0.5
        ),
        lambda value: value["setup_receipt"].__setitem__(
            "experiment_executed", True
        ),
        lambda value: value["resource_and_claim_firewall"].__setitem__(
            "setup_artifact_authorizes_no_success_claim", False
        ),
    ),
)
def test_validator_rejects_semantic_tampering_even_if_rehashed(mutate) -> None:
    value = deepcopy(
        build_fisher_activation_chart_whole_model_experiment_setup()
    )
    mutate(value)
    value["setup_sha256"] = setup_contract._sha256(
        {key: item for key, item in value.items() if key != "setup_sha256"}
    )
    with pytest.raises(ValueError, match="frozen v1"):
        validate_fisher_activation_chart_whole_model_experiment_setup(value)


def test_validator_rejects_an_invalid_setup_hash() -> None:
    value = deepcopy(build_fisher_activation_chart_whole_model_experiment_setup())
    value["setup_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen v1"):
        validate_fisher_activation_chart_whole_model_experiment_setup(value)


def test_setup_cli_never_imports_or_loads_model_runtime(tmp_path: Path) -> None:
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "fisher-graph-gemma-fisher-activation-chart-setup = "
        '"fisher_graph_fisher_activation_chart_setup:main"'
    ) in pyproject
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not ({"torch", "transformers", "mlx"} & imported)
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "load_gemma3" not in source
    assert "AutoModel" not in source

    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        """
import builtins

_original_import = builtins.__import__
_blocked = {"mlx", "torch", "transformers"}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in _blocked:
        raise RuntimeError(f"blocked runtime import: {name}")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guarded_import
""".lstrip(),
        encoding="utf-8",
    )
    destination = tmp_path / "setup.json"
    environment = dict(os.environ)
    python_path = [str(tmp_path), str(_REPO_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisher_graph_fisher_activation_chart_setup",
            "--output",
            str(destination),
        ],
        cwd=_REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = load_fisher_activation_chart_whole_model_experiment_setup(
        destination
    )
    assert loaded == build_fisher_activation_chart_whole_model_experiment_setup()


def test_setup_writer_is_create_only_and_constructor_fails_closed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "setup.json"
    assert main(["--output", str(destination)]) == 0
    assert load_fisher_activation_chart_whole_model_experiment_setup(
        destination
    ) == build_fisher_activation_chart_whole_model_experiment_setup()
    assert main(["--check", "--output", str(destination)]) == 0
    with pytest.raises(FileExistsError):
        write_fisher_activation_chart_whole_model_experiment_setup(destination)
    with pytest.raises(ValueError, match="chart_count"):
        FisherActivationChartWholeModelExperimentSetup(chart_count=0)
    with pytest.raises(ValueError, match="rank exceeds"):
        FisherActivationChartWholeModelExperimentSetup(chart_rank=641)


def test_checked_in_setup_matches_builder_if_present() -> None:
    source = _REPO_ROOT / DEFAULT_OUTPUT
    if source.exists():
        assert load_fisher_activation_chart_whole_model_experiment_setup(
            source
        ) == build_fisher_activation_chart_whole_model_experiment_setup()
