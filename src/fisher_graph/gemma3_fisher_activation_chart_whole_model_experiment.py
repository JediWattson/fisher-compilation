"""Runnable bounded Fisher/activation-chart whole-model Gemma experiment.

The default model-facing path reuses the existing progressive Fisher capture,
factorized chart fit, candidate-trajectory transport fit, and
``Gemma3CrossLayerChartGraphExecutor`` runtime for all 18 MLP mutations.  The
tied lexical boundary, attention, residual wiring, and normalizations remain
exact native operations in this bounded overlay arm.

This is not the lexical weight-SVD ablation and not the unrestricted
layer-agnostic primary.  The callback orchestration seam below makes the
capture -> chart -> candidate trajectory -> transport -> evaluation lineage
independently testable without loading a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from .gemma3_cross_layer_chart_transport_model_experiment import (
    CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION,
    CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA,
    DEFAULT_OUTPUT as _TRANSPORT_DEFAULT_OUTPUT,
    run_gemma3_cross_layer_chart_transport_model_experiment,
)
from .gemma3_cross_layer_chart_transport_protocol import (
    DEFAULT_PROTOCOL_OUTPUT as DEFAULT_TRANSPORT_PROTOCOL,
    PINNED_DEVELOPMENT_PANEL_PATH as DEFAULT_PANEL_PATH,
    PINNED_DEVELOPMENT_PANEL_SHA256,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION as DEFAULT_REVISION,
    build_cross_layer_chart_transport_protocol,
)
from .gemma3_fisher_activation_chart_whole_model_experiment_setup import (
    DEFAULT_OUTPUT as DEFAULT_SETUP_PATH,
    FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID,
    build_fisher_activation_chart_whole_model_experiment_setup,
    load_fisher_activation_chart_whole_model_experiment_setup,
    validate_fisher_activation_chart_whole_model_experiment_setup,
)
from .gemma3_layer_agnostic_whole_model_compression_protocol import (
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
)


FISHER_ACTIVATION_CHART_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_fisher_activation_chart_whole_model_experiment.v1"
)
FISHER_ACTIVATION_CHART_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "fisher-activation-chart-whole-model-bounded-v1.json"
)
_HASH_DOMAIN = b"fisher_graph:fisher-activation-chart-worker:v1\0"
_STAGE_HASH_DOMAIN = b"fisher_graph:fisher-activation-chart-stage:v1\0"
_TRANSPORT_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-cross-layer-chart-transport-report:v1\0"
)
_PHASES = (
    "capture_teacher_fisher_rows",
    "fit_chart_generators",
    "compile_first_candidate",
    "recapture_candidate_trajectory",
    "fit_candidate_transports",
    "compile_frozen_candidate",
    "evaluate_controls",
    "qualify",
)
_FUTURE_SPLIT_ROLES = ("A_fit", "A_select", "B_guard", "C_assessment")
_NATIVE_MLP_SCALARS = 18 * 3 * 640 * 2048
_FACTOR_CHART_LAYER_SCALARS = (
    2048 * 640
    + 2048
    + 640 * 2048
    + 8 * 640
    + 8 * 640 * 16
    + 8
    + 8 * 17 * 4
    + 8 * 4 * 2048
)
_TRANSPORT_EDGE_SCALARS = 4 * 4


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _json_safe_mapping(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    try:
        safe = json.loads(_canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} metadata must be finite and JSON-safe") from error
    if not isinstance(safe, dict):
        raise TypeError(f"{label} metadata must be a mapping")
    return safe


@dataclass(frozen=True, slots=True)
class FisherActivationChartStageArtifact:
    """Opaque runtime payload plus authenticated JSON-safe stage metadata."""

    phase: str
    payload: object
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("stage artifact phase is not frozen")
        _json_safe_mapping(self.metadata, label=self.phase)

    @property
    def artifact_sha256(self) -> str:
        metadata = _json_safe_mapping(self.metadata, label=self.phase)
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN
            + self.phase.encode("ascii")
            + b"\0"
            + _canonical_json_bytes(metadata)
        ).hexdigest()

    def receipt(self, *, input_artifact_sha256s: Sequence[str]) -> dict[str, object]:
        return {
            "phase": self.phase,
            "input_artifact_sha256s": list(input_artifact_sha256s),
            "metadata": _json_safe_mapping(self.metadata, label=self.phase),
            "artifact_sha256": self.artifact_sha256,
            "payload_serialized_in_report": False,
        }


CaptureCallback = Callable[
    [Mapping[str, object], object], FisherActivationChartStageArtifact
]
UnaryStageCallback = Callable[
    [Mapping[str, object], FisherActivationChartStageArtifact],
    FisherActivationChartStageArtifact,
]
RecaptureCallback = Callable[
    [Mapping[str, object], FisherActivationChartStageArtifact, object],
    FisherActivationChartStageArtifact,
]
TransportCallback = Callable[
    [
        Mapping[str, object],
        FisherActivationChartStageArtifact,
        FisherActivationChartStageArtifact,
    ],
    FisherActivationChartStageArtifact,
]
CompileFrozenCallback = TransportCallback
EvaluateCallback = Callable[
    [
        Mapping[str, object],
        FisherActivationChartStageArtifact,
        Mapping[str, object],
    ],
    FisherActivationChartStageArtifact,
]


@dataclass(frozen=True, slots=True)
class FisherActivationChartWorkerDependencies:
    """Injectable stage callbacks for synthetic and alternate workers."""

    capture_teacher_fisher_rows: CaptureCallback
    fit_chart_generators: UnaryStageCallback
    compile_first_candidate: UnaryStageCallback
    recapture_candidate_trajectory: RecaptureCallback
    fit_candidate_transports: TransportCallback
    compile_frozen_candidate: CompileFrozenCallback
    evaluate_controls: EvaluateCallback
    qualify: UnaryStageCallback

    def __post_init__(self) -> None:
        for name in _PHASES:
            if not callable(getattr(self, name)):
                raise TypeError(f"worker dependency {name} must be callable")


def _require_stage(
    value: object,
    *,
    phase: str,
) -> FisherActivationChartStageArtifact:
    if not isinstance(value, FisherActivationChartStageArtifact):
        raise TypeError(f"{phase} callback must return a stage artifact")
    if value.phase != phase:
        raise ValueError(f"{phase} callback returned the wrong phase")
    return value


def _orchestrate_fisher_activation_chart_worker(
    *,
    setup: Mapping[str, object],
    splits: Mapping[str, object],
    dependencies: FisherActivationChartWorkerDependencies,
) -> dict[str, object]:
    """Exercise the deferred full-protocol seam with split isolation.

    This injectable DAG is not the default model-facing development worker.
    It exists so the future A-fit/select/guard/assessment protocol can be
    tested without conflating it with the two-way A-fit/A-eval overlay below.
    """

    frozen_setup = validate_fisher_activation_chart_whole_model_experiment_setup(
        setup
    )
    if set(splits) != set(_FUTURE_SPLIT_ROLES):
        raise ValueError("worker splits must provide the four frozen roles")
    if not isinstance(dependencies, FisherActivationChartWorkerDependencies):
        raise TypeError("dependencies have the wrong type")
    fit_batches = splits["A_fit"]
    evaluation_splits = {
        role: splits[role] for role in _FUTURE_SPLIT_ROLES if role != "A_fit"
    }
    receipts: list[dict[str, object]] = []

    capture = _require_stage(
        dependencies.capture_teacher_fisher_rows(frozen_setup, fit_batches),
        phase="capture_teacher_fisher_rows",
    )
    receipts.append(capture.receipt(input_artifact_sha256s=()))
    charts = _require_stage(
        dependencies.fit_chart_generators(frozen_setup, capture),
        phase="fit_chart_generators",
    )
    receipts.append(
        charts.receipt(input_artifact_sha256s=(capture.artifact_sha256,))
    )
    first_candidate = _require_stage(
        dependencies.compile_first_candidate(frozen_setup, charts),
        phase="compile_first_candidate",
    )
    receipts.append(
        first_candidate.receipt(
            input_artifact_sha256s=(charts.artifact_sha256,)
        )
    )
    trajectory = _require_stage(
        dependencies.recapture_candidate_trajectory(
            frozen_setup,
            first_candidate,
            fit_batches,
        ),
        phase="recapture_candidate_trajectory",
    )
    receipts.append(
        trajectory.receipt(
            input_artifact_sha256s=(first_candidate.artifact_sha256,)
        )
    )
    transports = _require_stage(
        dependencies.fit_candidate_transports(
            frozen_setup,
            charts,
            trajectory,
        ),
        phase="fit_candidate_transports",
    )
    receipts.append(
        transports.receipt(
            input_artifact_sha256s=(
                charts.artifact_sha256,
                trajectory.artifact_sha256,
            )
        )
    )
    frozen_candidate = _require_stage(
        dependencies.compile_frozen_candidate(
            frozen_setup,
            charts,
            transports,
        ),
        phase="compile_frozen_candidate",
    )
    receipts.append(
        frozen_candidate.receipt(
            input_artifact_sha256s=(
                charts.artifact_sha256,
                transports.artifact_sha256,
            )
        )
    )
    evaluation = _require_stage(
        dependencies.evaluate_controls(
            frozen_setup,
            frozen_candidate,
            evaluation_splits,
        ),
        phase="evaluate_controls",
    )
    receipts.append(
        evaluation.receipt(
            input_artifact_sha256s=(frozen_candidate.artifact_sha256,)
        )
    )
    qualification = _require_stage(
        dependencies.qualify(frozen_setup, evaluation),
        phase="qualify",
    )
    receipts.append(
        qualification.receipt(
            input_artifact_sha256s=(evaluation.artifact_sha256,)
        )
    )
    if [receipt["phase"] for receipt in receipts] != list(_PHASES):
        raise RuntimeError("worker phase order drifted")
    evaluation_metadata = _json_safe_mapping(
        evaluation.metadata, label="evaluate_controls"
    )
    qualification_metadata = _json_safe_mapping(
        qualification.metadata, label="qualify"
    )
    resources = evaluation_metadata.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("evaluation metadata must close bounded resources")
    resource_accounting = _validate_resource_accounting(resources)
    forbidden_true_claims = (
        "qualified",
        "compression_success_claimed",
        "compression_claim_authorized",
        "source_free_runtime_established",
        "latency_claimed",
    )
    if any(qualification_metadata.get(name) is True for name in forbidden_true_claims):
        raise ValueError("bounded qualification attempted an unauthorized claim")
    qualification_metadata.update(
        {
            "qualified": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "source_free_runtime_established": False,
            "latency_claimed": False,
        }
    )
    report: dict[str, object] = {
        "schema": FISHER_ACTIVATION_CHART_EXPERIMENT_SCHEMA,
        "format_version": FISHER_ACTIVATION_CHART_EXPERIMENT_FORMAT_VERSION,
        "execution_kind": "future_full_protocol_injectable_orchestration_seam",
        "bounded_arm_id": FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID,
        "setup_sha256": frozen_setup["setup_sha256"],
        "phase_receipts": receipts,
        "evaluation_metadata": evaluation_metadata,
        "resource_accounting": resource_accounting,
        "qualification": qualification_metadata,
        "split_firewall": {
            "fit_roles_seen_by_fit_callbacks": ["A_fit"],
            "evaluation_roles_seen_by_evaluation_callback": list(
                evaluation_splits
            ),
            "A_fit_seen_by_evaluation_callback": False,
        },
        "scientific_status": {
            "bounded_partial_arm": True,
            "future_protocol_test_seam_only": True,
            "default_model_facing_worker_executed": False,
            "unrestricted_primary_executed": False,
            "compression_success_claimed": False,
            "source_free_runtime_claimed": False,
            "latency_claimed": False,
        },
    }
    report["report_sha256"] = _sha256(report)
    return validate_fisher_activation_chart_whole_model_experiment_report(report)


def _validate_resource_accounting(
    value: Mapping[str, object],
) -> dict[str, object]:
    safe = _json_safe_mapping(value, label="resources")
    source = safe.get("source_unique_parameters")
    candidate = safe.get("candidate_unique_parameters")
    compiled = safe.get("compiled_graph_coefficient_scalars")
    resident = safe.get("physical_resident_overlay_scalars")
    all_stacks = safe.get("all_fitted_stack_coefficient_scalars")
    worker_resident = safe.get("development_worker_resident_lower_bound_scalars")
    removed = safe.get("removed_parameters")
    fraction = safe.get("parameter_reduction_fraction")
    if (
        type(source) is not int
        or source != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or type(candidate) is not int
        or candidate <= 0
        or type(compiled) is not int
        or compiled <= 0
        or type(resident) is not int
        or resident != source + compiled
        or type(all_stacks) is not int
        or all_stacks < compiled
        or type(worker_resident) is not int
        or worker_resident != source + all_stacks
        or type(removed) is not int
        or removed != source - candidate
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or float(fraction) != removed / source
        or safe.get("inventory_closes") is not True
        or safe.get("copied_exact_coefficients_counted") is not True
        or safe.get("tied_lexical_parameters_counted_once") is not True
        or safe.get("physical_resident_overlay_inventory_closes") is not True
        or safe.get("physical_resident_scope")
        != "selected_arm_only_not_development_worker_peak"
        or safe.get("projected_logical_candidate_float32_bytes") != 4 * candidate
        or safe.get("physical_resident_overlay_float32_bytes") != 4 * resident
        or safe.get("development_worker_resident_lower_bound_float32_bytes")
        != 4 * worker_resident
        or safe.get("serialized_candidate_artifact_written") is not False
        or safe.get("whole_model_MAC_inventory_closes") is not False
        or safe.get("resident_memory_reduction_claimed") is not False
    ):
        raise ValueError("bounded resource accounting does not close")
    return safe


def _transport_report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _TRANSPORT_REPORT_HASH_DOMAIN + _canonical_json_bytes(dict(value))
    ).hexdigest()


def _all_native_projection_counts_are_zero(
    receipt: Mapping[str, object],
) -> bool:
    counts = receipt.get("native_source_projection_call_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        str(index) for index in range(18)
    }:
        return False
    for layer_counts in counts.values():
        if not isinstance(layer_counts, Mapping) or set(layer_counts) != {
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            return False
        if any(value != 0 for value in layer_counts.values()):
            return False
    return True


def _validate_bounded_transport_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the delegated development report before adding conclusions."""

    nested = _json_safe_mapping(value, label="bounded transport report")
    if (
        nested.get("schema") != CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA
        or nested.get("format_version")
        != CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION
    ):
        raise ValueError("delegated transport report schema is invalid")
    digest = nested.get("report_sha256")
    without_digest = {
        key: item for key, item in nested.items() if key != "report_sha256"
    }
    if digest != _transport_report_sha256(without_digest):
        raise ValueError("delegated transport report SHA-256 is invalid")
    expected_protocol = build_cross_layer_chart_transport_protocol()
    if nested.get("protocol_sha256") != expected_protocol.get(
        "protocol_sha256"
    ):
        raise ValueError("delegated transport protocol lineage drifted")

    status = nested.get("scientific_status")
    scope = nested.get("execution_scope")
    model = nested.get("model")
    data = nested.get("data")
    configuration = nested.get("configuration")
    if not all(
        isinstance(section, Mapping)
        for section in (status, scope, model, data, configuration)
    ):
        raise ValueError("delegated transport report sections are incomplete")
    assert isinstance(status, Mapping)
    assert isinstance(scope, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(configuration, Mapping)
    required_status = {
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
    }
    if any(
        status.get(key) is not expected
        for key, expected in required_status.items()
    ):
        raise ValueError("delegated transport scientific status drifted")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("revision") != DEFAULT_REVISION
        or model.get("device") != "cpu"
        or model.get("dtype") != "float32"
        or model.get("source_whole_model_parameters")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    ):
        raise ValueError("delegated transport model receipt drifted")
    if (
        data.get("panel_sha256") != PINNED_DEVELOPMENT_PANEL_SHA256
        or data.get("prompt_disjoint") is not True
        or data.get("family_disjoint") is not False
        or data.get("family_count") != 6
        or data.get("fit_example_count") != 30
        or data.get("evaluation_example_count") != 30
    ):
        raise ValueError("delegated transport data split drifted")
    expected_configuration = {
        "generator_rank": 4,
        "chart_temperature": 1.0,
        "affine_relative_ridge": 1.0e-3,
        "local_chart_relative_ridge": 1.0e-2,
        "maximum_incoming_charts_per_target": 2,
        "transport_ridge": 1.0e-4,
        "row_shuffled_control_executed": True,
        "same_token_adjacent_layer_only": True,
    }
    if dict(configuration) != expected_configuration:
        raise ValueError("delegated transport configuration drifted")
    expected_arms = [
        "native",
        "progressive_independent_factorized_chart",
        "progressive_factorized_chart_transport",
        "transport_zero_edge_control",
        "transport_row_shuffled_negative_control",
    ]
    if scope.get("executed_arms") != expected_arms:
        raise ValueError("delegated transport control coverage drifted")
    if (
        scope.get("compression_criterion_evaluated") is not False
        or scope.get("causal_confirmation_evaluated") is not False
        or scope.get("unexecuted_protocol_arms")
        != [
            "exact_all_mode_graph",
            "one_shot_independent_dense_chart_reproduction",
            "budget_matched_factorized_chart_transport",
        ]
    ):
        raise ValueError("delegated transport claim scope drifted")

    fit_metrics = nested.get("progressive_layer_fit")
    freeze_receipts = nested.get("progressive_freeze_receipts")
    evaluations = nested.get("full_model_evaluation")
    executions = nested.get("execution_receipts")
    resources = nested.get("resources")
    fit_arms = expected_arms[1:3] + [expected_arms[4]]
    evaluated_arms = expected_arms[1:]
    if (
        not isinstance(fit_metrics, Mapping)
        or set(fit_metrics) != set(fit_arms)
        or not isinstance(freeze_receipts, Mapping)
        or set(freeze_receipts) != set(fit_arms)
        or not isinstance(evaluations, Mapping)
        or set(evaluations) != set(evaluated_arms)
        or not isinstance(executions, Mapping)
        or set(executions) != set(evaluated_arms)
        or not isinstance(resources, Mapping)
        or set(resources) != set(fit_arms)
    ):
        raise ValueError("delegated transport arm receipts drifted")
    for arm in fit_arms:
        layers = fit_metrics[arm]
        receipts = freeze_receipts[arm]
        if (
            not isinstance(layers, Mapping)
            or set(layers) != {str(index) for index in range(18)}
            or not isinstance(receipts, list)
            or len(receipts) != 18
            or [receipt.get("layer_ordinal") for receipt in receipts]
            != list(range(18))
        ):
            raise ValueError("delegated progressive layer coverage drifted")
    for arm in evaluated_arms:
        receipt = executions[arm]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("compiled_layer_calls_per_batch") != 18
            or not _all_native_projection_counts_are_zero(receipt)
        ):
            raise ValueError("delegated source-call receipt drifted")
    for arm in fit_arms:
        arm_resource = resources[arm]
        execution = executions[arm]
        if not isinstance(arm_resource, Mapping) or not isinstance(
            execution, Mapping
        ):
            raise ValueError("delegated resource receipt is unavailable")
        edge_count = execution.get("transport_edge_count")
        if type(edge_count) is not int or edge_count < 0:
            raise ValueError("delegated transport edge count is invalid")
        expected_compiled = (
            18 * _FACTOR_CHART_LAYER_SCALARS
            + edge_count * _TRANSPORT_EDGE_SCALARS
        )
        candidate_scalars = arm_resource.get(
            "logical_candidate_whole_model_scalars"
        )
        if (
            arm_resource.get("logical_compiled_coefficient_scalars")
            != expected_compiled
            or type(candidate_scalars) is not int
            or candidate_scalars
            != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            - _NATIVE_MLP_SCALARS
            + expected_compiled
            or arm_resource.get("native_target_mlp_parameter_scalars")
            != _NATIVE_MLP_SCALARS
            or arm_resource.get("logical_whole_model_reduction_fraction")
            != (
                (PINNED_SOURCE_UNIQUE_PARAMETER_COUNT - candidate_scalars)
                / PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            )
            or arm_resource.get("shadow_runtime_retains_native_source_weights")
            is not True
            or arm_resource.get(
                "source_free_serialized_candidate_materialized"
            )
            is not False
            or arm_resource.get("realized_kernel_latency_measured") is not False
            or arm_resource.get("realized_resident_memory_reduction_claimed")
            is not False
            or arm_resource.get("compression_success_claimed") is not False
        ):
            raise ValueError("delegated selected resource ledger drifted")
    if executions["progressive_factorized_chart_transport"].get(
        "transport_edge_count"
    ) != executions["transport_zero_edge_control"].get("transport_edge_count"):
        raise ValueError("nested zero-edge control topology drifted")
    return nested


def _projected_overlay_resource_accounting(
    bounded: Mapping[str, object],
) -> dict[str, object]:
    model = bounded.get("model")
    resources = bounded.get("resources")
    if not isinstance(model, Mapping) or not isinstance(resources, Mapping):
        raise ValueError("bounded transport report omitted model resources")
    selected = resources.get("progressive_factorized_chart_transport")
    source_parameters = model.get("source_whole_model_parameters")
    if not isinstance(selected, Mapping) or type(source_parameters) is not int:
        raise ValueError("bounded transport resource arm is unavailable")
    candidate_parameters = selected.get("logical_candidate_whole_model_scalars")
    compiled_parameters = selected.get("logical_compiled_coefficient_scalars")
    if type(candidate_parameters) is not int or type(compiled_parameters) is not int:
        raise ValueError("bounded transport candidate count is unavailable")
    removed_parameters = source_parameters - candidate_parameters
    physical_resident_parameters = source_parameters + compiled_parameters
    all_fitted_stack_parameters = sum(
        int(resource["logical_compiled_coefficient_scalars"])
        for resource in resources.values()
        if isinstance(resource, Mapping)
    )
    worker_resident_parameters = source_parameters + all_fitted_stack_parameters
    return _validate_resource_accounting(
        {
            "source_unique_parameters": source_parameters,
            "candidate_unique_parameters": candidate_parameters,
            "compiled_graph_coefficient_scalars": compiled_parameters,
            "physical_resident_overlay_scalars": physical_resident_parameters,
            "all_fitted_stack_coefficient_scalars": (
                all_fitted_stack_parameters
            ),
            "development_worker_resident_lower_bound_scalars": (
                worker_resident_parameters
            ),
            "removed_parameters": removed_parameters,
            "parameter_reduction_fraction": (
                removed_parameters / source_parameters
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
                4 * physical_resident_parameters
            ),
            "development_worker_resident_lower_bound_float32_bytes": (
                4 * worker_resident_parameters
            ),
            "serialized_candidate_artifact_written": False,
            "whole_model_MAC_inventory_closes": False,
            "resident_memory_reduction_claimed": False,
        }
    )


def _wrap_bounded_transport_report(
    *,
    setup: Mapping[str, object],
    transport_report: Mapping[str, object],
) -> dict[str, object]:
    """Bind the existing real Fisher/chart worker to the new bounded scope."""

    setup_receipt = validate_fisher_activation_chart_whole_model_experiment_setup(
        setup
    )
    bounded = _validate_bounded_transport_report(transport_report)
    resource_accounting = _projected_overlay_resource_accounting(bounded)
    report: dict[str, object] = {
        "schema": FISHER_ACTIVATION_CHART_EXPERIMENT_SCHEMA,
        "format_version": FISHER_ACTIVATION_CHART_EXPERIMENT_FORMAT_VERSION,
        "execution_kind": "model_facing_bounded_progressive_chart_transport",
        "bounded_arm_id": FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID,
        "setup_sha256": setup_receipt["setup_sha256"],
        "runtime_scope": {
            "native_tied_lexical_boundary_preserved": True,
            "tied_lexical_parameters_counted_once": True,
            "runtime_tied_storage_alias_reverified": False,
            "native_attention_residual_and_norm_path_preserved": True,
            "fisher_chart_MLP_overlay_count": 18,
            "reference_transformer_event_count": 39,
            "independent_39_event_executor_executed": False,
            "runtime_is_native_model_overlay": True,
            "native_MLP_projection_calls_during_candidate": 0,
            "chart_count": 8,
            "chart_coordinate_rank": 16,
            "generator_rank": 4,
            "transport_rank": 4,
            "maximum_incoming_charts_per_target": 2,
            "batch_size": 4,
            "deterministic_seed": 1729,
            "source_free_runtime_established": False,
        },
        "bounded_transport_report": bounded,
        "resource_accounting": resource_accounting,
        "qualification": {
            "qualified": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "source_free_runtime_established": False,
            "latency_claimed": False,
            "failure_reasons": [
                "bounded_partial_arm",
                "same_family_prompt_disjoint_development_split_only",
                "serialized_source_free_candidate_not_materialized",
                "physical_overlay_resident_memory_exceeds_source",
                "whole_model_MAC_inventory_not_closed",
            ],
        },
        "scientific_status": {
            "bounded_partial_arm": True,
            "actual_model_facing_worker_executed": True,
            "teacher_fisher_rows_captured": True,
            "activation_charts_fitted": True,
            "candidate_trajectory_transports_fitted": True,
            "chart_conditioned_candidate_executed": True,
            "exact_39_event_control_executed": False,
            "unrestricted_primary_executed": False,
            "family_disjoint_evaluation": False,
            "compression_success_claimed": False,
            "source_free_runtime_claimed": False,
            "latency_claimed": False,
        },
        "deferred": setup_receipt["deferred_from_bounded_arm"],
    }
    report["report_sha256"] = _sha256(report)
    return validate_fisher_activation_chart_whole_model_experiment_report(report)


def validate_fisher_activation_chart_whole_model_experiment_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate hashes, lineage, resources, and bounded claims."""

    if not isinstance(value, Mapping):
        raise TypeError("Fisher activation-chart report must be a mapping")
    report = _json_safe_mapping(value, label="report")
    if (
        report.get("schema") != FISHER_ACTIVATION_CHART_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != FISHER_ACTIVATION_CHART_EXPERIMENT_FORMAT_VERSION
        or report.get("bounded_arm_id")
        != FISHER_ACTIVATION_CHART_BOUNDED_ARM_ID
    ):
        raise ValueError("unsupported Fisher activation-chart report")
    expected_setup = build_fisher_activation_chart_whole_model_experiment_setup()
    if report.get("setup_sha256") != expected_setup.get("setup_sha256"):
        raise ValueError("Fisher activation-chart setup lineage drifted")
    resources = report.get("resource_accounting")
    qualification = report.get("qualification")
    status = report.get("scientific_status")
    if not isinstance(resources, Mapping) or not isinstance(
        qualification, Mapping
    ) or not isinstance(status, Mapping):
        raise ValueError("Fisher activation-chart report sections are incomplete")
    if _validate_resource_accounting(resources) != dict(resources):
        raise ValueError("Fisher activation-chart resource receipt drifted")
    required_false = (
        "qualified",
        "compression_success_claimed",
        "compression_claim_authorized",
        "source_free_runtime_established",
        "latency_claimed",
    )
    if any(qualification.get(name) is not False for name in required_false):
        raise ValueError("bounded qualification claim boundary is invalid")
    for name in (
        "compression_success_claimed",
        "source_free_runtime_claimed",
        "latency_claimed",
    ):
        if status.get(name) is not False:
            raise ValueError("bounded scientific claim boundary is invalid")

    execution_kind = report.get("execution_kind")
    if execution_kind == "future_full_protocol_injectable_orchestration_seam":
        receipts = report.get("phase_receipts")
        firewall = report.get("split_firewall")
        if (
            not isinstance(receipts, list)
            or len(receipts) != len(_PHASES)
            or not isinstance(firewall, Mapping)
            or firewall.get("fit_roles_seen_by_fit_callbacks") != ["A_fit"]
            or firewall.get("evaluation_roles_seen_by_evaluation_callback")
            != ["A_select", "B_guard", "C_assessment"]
            or firewall.get("A_fit_seen_by_evaluation_callback") is not False
        ):
            raise ValueError("worker phase firewall is invalid")
        artifact_hashes: list[str] = []
        expected_inputs: tuple[tuple[int, ...], ...] = (
            (),
            (0,),
            (1,),
            (2,),
            (1, 3),
            (1, 4),
            (5,),
            (6,),
        )
        for index, (phase, receipt) in enumerate(
            zip(_PHASES, receipts, strict=True)
        ):
            if not isinstance(receipt, Mapping) or receipt.get("phase") != phase:
                raise ValueError("worker phase receipt order is invalid")
            metadata = receipt.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("worker phase metadata is unavailable")
            digest = hashlib.sha256(
                _STAGE_HASH_DOMAIN
                + phase.encode("ascii")
                + b"\0"
                + _canonical_json_bytes(dict(metadata))
            ).hexdigest()
            inputs = [artifact_hashes[position] for position in expected_inputs[index]]
            if (
                receipt.get("artifact_sha256") != digest
                or receipt.get("input_artifact_sha256s") != inputs
                or receipt.get("payload_serialized_in_report") is not False
            ):
                raise ValueError("worker phase artifact lineage is invalid")
            artifact_hashes.append(digest)
    elif execution_kind == "model_facing_bounded_progressive_chart_transport":
        scope = report.get("runtime_scope")
        bounded = report.get("bounded_transport_report")
        if (
            not isinstance(scope, Mapping)
            or not isinstance(bounded, Mapping)
            or scope.get("native_tied_lexical_boundary_preserved") is not True
            or scope.get("tied_lexical_parameters_counted_once") is not True
            or scope.get("runtime_tied_storage_alias_reverified") is not False
            or scope.get("native_attention_residual_and_norm_path_preserved")
            is not True
            or scope.get("fisher_chart_MLP_overlay_count") != 18
            or scope.get("reference_transformer_event_count") != 39
            or scope.get("independent_39_event_executor_executed") is not False
            or scope.get("runtime_is_native_model_overlay") is not True
            or scope.get("native_MLP_projection_calls_during_candidate") != 0
            or scope.get("chart_count") != 8
            or scope.get("chart_coordinate_rank") != 16
            or scope.get("generator_rank") != 4
            or scope.get("transport_rank") != 4
            or scope.get("maximum_incoming_charts_per_target") != 2
            or scope.get("batch_size") != 4
            or scope.get("deterministic_seed") != 1729
            or scope.get("source_free_runtime_established") is not False
        ):
            raise ValueError("bounded model-facing runtime scope is invalid")
        validated_bounded = _validate_bounded_transport_report(bounded)
        if validated_bounded != dict(bounded):
            raise ValueError("nested bounded worker receipt drifted")
        if _projected_overlay_resource_accounting(validated_bounded) != dict(
            resources
        ):
            raise ValueError("bounded projected overlay accounting drifted")
        expected_status = {
            "bounded_partial_arm": True,
            "actual_model_facing_worker_executed": True,
            "teacher_fisher_rows_captured": True,
            "activation_charts_fitted": True,
            "candidate_trajectory_transports_fitted": True,
            "chart_conditioned_candidate_executed": True,
            "exact_39_event_control_executed": False,
            "unrestricted_primary_executed": False,
            "family_disjoint_evaluation": False,
            "compression_success_claimed": False,
            "source_free_runtime_claimed": False,
            "latency_claimed": False,
        }
        if dict(status) != expected_status:
            raise ValueError("bounded model-facing scientific status drifted")
        expected_qualification = {
            "qualified": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "source_free_runtime_established": False,
            "latency_claimed": False,
            "failure_reasons": [
                "bounded_partial_arm",
                "same_family_prompt_disjoint_development_split_only",
                "serialized_source_free_candidate_not_materialized",
                "physical_overlay_resident_memory_exceeds_source",
                "whole_model_MAC_inventory_not_closed",
            ],
        }
        if dict(qualification) != expected_qualification:
            raise ValueError("bounded model-facing qualification drifted")
    else:
        raise ValueError("unknown Fisher activation-chart execution kind")

    digest = report.get("report_sha256")
    without_digest = {
        key: item for key, item in report.items() if key != "report_sha256"
    }
    if digest != _sha256(without_digest):
        raise ValueError("Fisher activation-chart report SHA-256 is invalid")
    return report


def load_fisher_activation_chart_whole_model_experiment_report(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Fisher activation-chart report is invalid JSON") from error
    return validate_fisher_activation_chart_whole_model_experiment_report(value)


def run_gemma3_fisher_activation_chart_whole_model_experiment(
    *,
    revision: str = DEFAULT_REVISION,
    setup_path: Path | str = DEFAULT_SETUP_PATH,
    transport_protocol_path: Path | str = DEFAULT_TRANSPORT_PROTOCOL,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
    generator_rank: int = 4,
    maximum_incoming_charts_per_target: int = 2,
    transport_ridge: float = 1.0e-4,
    include_row_shuffled_control: bool = True,
) -> dict[str, object]:
    """Invoke the existing real full-stack Fisher/chart transport worker."""

    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("Fisher activation-chart output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a chart worker report")
    setup = load_fisher_activation_chart_whole_model_experiment_setup(setup_path)
    hyperparameters = setup.get("frozen_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Fisher activation-chart setup omitted hyperparameters")
    if (
        generator_rank != hyperparameters.get("generator_rank")
        or maximum_incoming_charts_per_target
        != hyperparameters.get("maximum_incoming_charts_per_target")
        or transport_ridge != hyperparameters.get("transport_relative_ridge")
        or hyperparameters.get("transport_rank") != generator_rank
        or hyperparameters.get("chart_count") != 8
        or hyperparameters.get("chart_rank") != 16
        or hyperparameters.get("affine_relative_ridge") != 1.0e-3
        or hyperparameters.get("local_chart_relative_ridge") != 1.0e-2
        or hyperparameters.get("chart_temperature") != 1.0
        or batch_size != hyperparameters.get("batch_size")
        or include_row_shuffled_control is not True
    ):
        raise ValueError("worker arguments differ from the frozen setup")
    with tempfile.TemporaryDirectory(
        prefix="fisher-activation-chart-worker-"
    ) as temporary:
        transport_output = Path(temporary) / _TRANSPORT_DEFAULT_OUTPUT.name
        bounded = run_gemma3_cross_layer_chart_transport_model_experiment(
            revision=revision,
            protocol_path=transport_protocol_path,
            output=transport_output,
            panel_path=panel_path,
            model_id=model_id,
            cache_dir=cache_dir,
            device_name=device_name,
            dtype=dtype,
            batch_size=batch_size,
            generator_rank=generator_rank,
            maximum_incoming_charts_per_target=(
                maximum_incoming_charts_per_target
            ),
            transport_ridge=transport_ridge,
            include_row_shuffled_control=include_row_shuffled_control,
        )
    report = _wrap_bounded_transport_report(
        setup=setup,
        transport_report=bounded,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return load_fisher_activation_chart_whole_model_experiment_report(
        destination
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded all-MLP Gemma Fisher/activation-chart transport "
            "experiment."
        )
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument(
        "--transport-protocol",
        type=Path,
        default=DEFAULT_TRANSPORT_PROTOCOL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generator-rank", type=int, default=4)
    parser.add_argument("--maximum-incoming-charts", type=int, default=2)
    parser.add_argument("--transport-ridge", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    run_gemma3_fisher_activation_chart_whole_model_experiment(
        revision=args.revision,
        setup_path=args.setup,
        transport_protocol_path=args.transport_protocol,
        output=args.output,
        panel_path=args.panel_path,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        generator_rank=args.generator_rank,
        maximum_incoming_charts_per_target=args.maximum_incoming_charts,
        transport_ridge=args.transport_ridge,
        include_row_shuffled_control=True,
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "FISHER_ACTIVATION_CHART_EXPERIMENT_FORMAT_VERSION",
    "FISHER_ACTIVATION_CHART_EXPERIMENT_SCHEMA",
    "FisherActivationChartStageArtifact",
    "FisherActivationChartWorkerDependencies",
    "_orchestrate_fisher_activation_chart_worker",
    "load_fisher_activation_chart_whole_model_experiment_report",
    "main",
    "run_gemma3_fisher_activation_chart_whole_model_experiment",
    "validate_fisher_activation_chart_whole_model_experiment_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
