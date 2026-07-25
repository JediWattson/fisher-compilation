"""Independent verification for a saved associative-recall Fisher build."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from .associative import (
    AssociativeRecallMetrics,
    AssociativeRecallSplits,
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
    evaluate_associative_recall,
)
from .config import TransformerConfig
from .compiler import (
    load_runtime_manifest,
    manifest_from_legacy_runtime,
    open_verified_resource,
    runtime_manifest_bytes,
)
from .fused_executor import (
    FusedToyTransformer,
    FusedTwoLayerModalStack,
    LazyFusedTwoLayerModalStack,
    PackedTriangularFusedTwoLayerModalStack,
    load_fused_modal_stack,
    load_lazy_fused_modal_stack,
)
from .layers import LayerExecutor, TransformerBlock
from .modal_artifacts import (
    fused_executor_artifact_paths,
    modal_completion_artifact_paths,
    modal_executor_artifact_paths,
)
from .modal_completion import (
    ModalCompletionFitConfig,
    PositionConditionedCompletedModalGraphExecutor,
    PositionConditionedModalCompletion,
    PositionConditionedModalCompletionBottleneckExecutor,
    fit_local_modal_completion,
    load_position_modal_completion,
    make_mean_modal_completion,
)
from .modal_executor import (
    CausalModalMLPGraph,
    PositionConditionedModalBottleneckExecutor,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
    load_position_modal_executor,
)
from .modes import FisherModeBasis, load_fisher_build
from .model import ToyTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} is nonfinite")
    return result


def _assert_finite_tree(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, path)


def _assert_metrics_match(
    actual_value: object,
    expected: dict[str, object],
    *,
    path: str = "metrics",
) -> None:
    actual = _object(actual_value, path)
    if set(actual) != set(expected):
        raise ValueError(f"{path} metric fields mismatch")
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, tuple):
            values = _array(actual_value, f"{path}.{name}")
            if len(values) != len(expected_value):
                raise ValueError(f"{path}.{name} length mismatch")
            if any(
                not math.isclose(
                    _number(value, f"{path}.{name}"),
                    expected_item,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
                for value, expected_item in zip(
                    values, expected_value, strict=True
                )
            ):
                raise ValueError(f"{path}.{name} mismatch")
        elif isinstance(expected_value, float):
            if not math.isclose(
                _number(actual_value, f"{path}.{name}"),
                expected_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{path}.{name} mismatch")
        elif actual_value != expected_value:
            raise ValueError(f"{path}.{name} mismatch")


def _validate_metrics_structure(
    value: object,
    *,
    template: AssociativeRecallMetrics,
    samples: int,
    contexts: int,
    path: str,
) -> dict[str, object]:
    metrics = _object(value, path)
    expected = asdict(template)
    if set(metrics) != set(expected):
        raise ValueError(f"{path} metric fields mismatch")
    if _integer(metrics.get("samples"), f"{path}.samples") != samples:
        raise ValueError(f"{path}.samples mismatch")
    if _integer(metrics.get("contexts"), f"{path}.contexts") != contexts:
        raise ValueError(f"{path}.contexts mismatch")

    hard_nll = _number(metrics.get("hard_nll"), f"{path}.hard_nll")
    if hard_nll < 0:
        raise ValueError(f"{path}.hard_nll is negative")
    bounded_names = (
        "answer_accuracy",
        "paired_context_accuracy",
        "minimum_query_accuracy",
        "minimum_value_accuracy",
        "mean_correct_probability",
    )
    for name in bounded_names:
        number = _number(metrics.get(name), f"{path}.{name}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{path}.{name} is outside [0, 1]")

    arrays: dict[str, list[float]] = {}
    for name, expected_value in expected.items():
        if not isinstance(expected_value, tuple):
            continue
        values = [
            _number(item, f"{path}.{name}[{index}]")
            for index, item in enumerate(
                _array(metrics.get(name), f"{path}.{name}")
            )
        ]
        if len(values) != len(expected_value):
            raise ValueError(f"{path}.{name} length mismatch")
        if any(not 0.0 <= item <= 1.0 for item in values):
            raise ValueError(f"{path}.{name} contains a value outside [0, 1]")
        arrays[name] = values

    minimum_pairs = (
        ("query_accuracies", "minimum_query_accuracy"),
        ("value_accuracies", "minimum_value_accuracy"),
    )
    for array_name, minimum_name in minimum_pairs:
        values = arrays[array_name]
        reported = _number(metrics.get(minimum_name), f"{path}.{minimum_name}")
        if not math.isclose(
            min(values),
            reported,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{path}.{minimum_name} mismatch")
    return metrics


def _validate_modal_fit(
    value: object,
    *,
    path: str,
    expected_parameters: int | None = None,
    expected_edges: int | None = None,
) -> tuple[dict[str, object], int]:
    fit = _object(value, path)
    best_step = _integer(fit.get("best_step"), f"{path}.best_step")
    if best_step <= 0:
        raise ValueError(f"{path}.best_step must be positive")
    for name in ("train_mse", "validation_mse"):
        if _number(fit.get(name), f"{path}.{name}") < 0:
            raise ValueError(f"{path}.{name} is negative")
    learned_parameters = _integer(
        fit.get("learned_parameters"),
        f"{path}.learned_parameters",
    )
    graph_edges = _integer(fit.get("graph_edges"), f"{path}.graph_edges")
    if learned_parameters <= 0 or graph_edges <= 0:
        raise ValueError(f"{path} parameter and edge counts must be positive")
    if (
        expected_parameters is not None
        and learned_parameters != expected_parameters
    ):
        raise ValueError(f"{path}.learned_parameters mismatch")
    if expected_edges is not None and graph_edges != expected_edges:
        raise ValueError(f"{path}.graph_edges mismatch")

    history = _array(fit.get("history"), f"{path}.history")
    if not history:
        raise ValueError(f"{path}.history cannot be empty")
    steps: list[int] = []
    validation_values: list[float] = []
    for index, value in enumerate(history):
        point_path = f"{path}.history[{index}]"
        point = _object(value, point_path)
        step = _integer(point.get("step"), f"{point_path}.step")
        batch_mse = _number(
            point.get("batch_mse"),
            f"{point_path}.batch_mse",
        )
        validation_mse = _number(
            point.get("validation_mse"),
            f"{point_path}.validation_mse",
        )
        if step <= 0 or batch_mse < 0 or validation_mse < 0:
            raise ValueError(f"{point_path} is invalid")
        steps.append(step)
        validation_values.append(validation_mse)
    if steps != sorted(set(steps)):
        raise ValueError(f"{path}.history steps are not strictly increasing")
    if best_step not in steps:
        raise ValueError(f"{path}.best_step is absent from history")
    minimum_index = validation_values.index(min(validation_values))
    if steps[minimum_index] != best_step:
        raise ValueError(f"{path}.best_step is not the validation minimum")
    if not math.isclose(
        _number(fit.get("validation_mse"), f"{path}.validation_mse"),
        min(validation_values),
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}.validation_mse is not the selected minimum")
    return fit, len(history)


def _json_normalized(value: object) -> object:
    """Normalize tuples and primitive containers as JSON would."""

    return json.loads(json.dumps(value, allow_nan=False))


def _validate_modal_fit_protocol(
    report_value: object,
    metadata_value: object,
) -> tuple[dict[str, object], dict[str, object]]:
    report_protocol = _object(
        report_value,
        "modal executor report fit_protocol",
    )
    metadata_protocol = _object(
        metadata_value,
        "modal executor metadata fit_protocol",
    )
    if _json_normalized(report_protocol) != _json_normalized(
        metadata_protocol
    ):
        raise ValueError("modal executor fit protocol metadata mismatch")
    expected_protocol_fields = {
        "fit_config",
        "coordinate_normalization",
        "standard_deviation_correction",
        "minimum_scale",
        "initialization_count",
        "selection_rule",
    }
    if set(report_protocol) != expected_protocol_fields:
        raise ValueError("modal executor fit protocol fields mismatch")

    fit_config = _object(
        report_protocol.get("fit_config"),
        "modal executor fit_protocol.fit_config",
    )
    expected_fit_fields = {
        "steps",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "evaluation_interval",
        "seed",
        "device",
        "minimum_scale",
    }
    if set(fit_config) != expected_fit_fields:
        raise ValueError("modal executor fit config fields mismatch")
    steps = _integer(
        fit_config.get("steps"),
        "modal executor fit config.steps",
    )
    batch_size = _integer(
        fit_config.get("batch_size"),
        "modal executor fit config.batch_size",
    )
    evaluation_interval = _integer(
        fit_config.get("evaluation_interval"),
        "modal executor fit config.evaluation_interval",
    )
    seed = _integer(
        fit_config.get("seed"),
        "modal executor fit config.seed",
    )
    learning_rate = _number(
        fit_config.get("learning_rate"),
        "modal executor fit config.learning_rate",
    )
    weight_decay = _number(
        fit_config.get("weight_decay"),
        "modal executor fit config.weight_decay",
    )
    fit_minimum_scale = _number(
        fit_config.get("minimum_scale"),
        "modal executor fit config.minimum_scale",
    )
    if (
        steps <= 0
        or batch_size <= 0
        or evaluation_interval <= 0
        or evaluation_interval > steps
        or seed < 0
        or learning_rate <= 0
        or weight_decay < 0
        or fit_minimum_scale <= 0
    ):
        raise ValueError("modal executor fit config is invalid")
    device_value = fit_config.get("device")
    if not isinstance(device_value, str) or not device_value:
        raise ValueError("modal executor fit config.device is invalid")
    try:
        torch.device(device_value)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "modal executor fit config.device is invalid"
        ) from error

    if (
        report_protocol.get("coordinate_normalization")
        != "per_position_mode_sample_std"
    ):
        raise ValueError("modal executor coordinate normalization mismatch")
    if _integer(
        report_protocol.get("standard_deviation_correction"),
        "modal executor fit protocol.standard_deviation_correction",
    ) != 1:
        raise ValueError(
            "modal executor standard-deviation correction mismatch"
        )
    protocol_minimum_scale = _number(
        report_protocol.get("minimum_scale"),
        "modal executor fit protocol.minimum_scale",
    )
    if (
        protocol_minimum_scale != 1e-4
        or fit_minimum_scale != protocol_minimum_scale
    ):
        raise ValueError("modal executor minimum scale mismatch")
    if _integer(
        report_protocol.get("initialization_count"),
        "modal executor fit protocol.initialization_count",
    ) != 1:
        raise ValueError("modal executor initialization count mismatch")
    if (
        report_protocol.get("selection_rule")
        != "smallest_validation_gate_passing_width_else_lowest_nll"
    ):
        raise ValueError("modal executor selection rule mismatch")
    return report_protocol, fit_config


def _validate_modal_history_schedule(
    fit_value: object,
    fit_config: dict[str, object],
    *,
    path: str,
) -> None:
    fit = _object(fit_value, path)
    history = _array(fit.get("history"), f"{path}.history")
    actual_steps = [
        _integer(
            _object(point, f"{path}.history[{index}]").get("step"),
            f"{path}.history[{index}].step",
        )
        for index, point in enumerate(history)
    ]
    steps = _integer(fit_config.get("steps"), "modal fit config.steps")
    interval = _integer(
        fit_config.get("evaluation_interval"),
        "modal fit config.evaluation_interval",
    )
    expected_steps = sorted(
        {1, steps, *range(interval, steps + 1, interval)}
    )
    if actual_steps != expected_steps:
        raise ValueError(f"{path}.history does not match fit protocol")


def _validate_bootstrap(
    value: object,
    *,
    contexts: int,
    path: str,
) -> None:
    bootstrap = _object(value, path)
    if _integer(bootstrap.get("contexts"), f"{path}.contexts") != contexts:
        raise ValueError(f"{path}.contexts mismatch")
    if _integer(
        bootstrap.get("bootstrap_samples"), f"{path}.bootstrap_samples"
    ) <= 0:
        raise ValueError(f"{path}.bootstrap_samples must be positive")
    _integer(bootstrap.get("seed"), f"{path}.seed")
    _number(
        bootstrap.get("mean_top_minus_bottom_delta_hard_nll"),
        f"{path}.mean_top_minus_bottom_delta_hard_nll",
    )
    interval = _array(
        bootstrap.get("confidence_interval_95"),
        f"{path}.confidence_interval_95",
    )
    if len(interval) != 2:
        raise ValueError(f"{path}.confidence_interval_95 must have two values")
    if _number(interval[0], f"{path}.confidence_interval_95[0]") > _number(
        interval[1], f"{path}.confidence_interval_95[1]"
    ):
        raise ValueError(f"{path}.confidence_interval_95 is reversed")
    fraction = _number(
        bootstrap.get("fraction_bootstrap_above_zero"),
        f"{path}.fraction_bootstrap_above_zero",
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{path}.fraction_bootstrap_above_zero is invalid")


def _validate_result(
    value: object,
    *,
    bases: dict[str, object],
    samples: int,
    contexts: int,
    path: str,
) -> dict[str, object]:
    result = _object(value, path)
    boundary = result.get("boundary")
    if boundary not in bases:
        raise ValueError(f"{path}.boundary is unknown")
    width = bases[str(boundary)].width  # type: ignore[attr-defined]
    indices = [
        _integer(item, f"{path}.mode_indices")
        for item in _array(result.get("mode_indices"), f"{path}.mode_indices")
    ]
    if len(indices) != len(set(indices)) or any(
        index < 0 or index >= width for index in indices
    ):
        raise ValueError(f"{path}.mode_indices is invalid")
    if _integer(result.get("mode_count"), f"{path}.mode_count") != len(indices):
        raise ValueError(f"{path}.mode_count mismatch")
    fraction = _number(
        result.get("suppression_fraction"),
        f"{path}.suppression_fraction",
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{path}.suppression_fraction is invalid")
    if _number(
        result.get("activation_delta_rms"),
        f"{path}.activation_delta_rms",
    ) < 0:
        raise ValueError(f"{path}.activation_delta_rms is negative")
    metrics = _object(result.get("metrics"), f"{path}.metrics")
    if _integer(metrics.get("samples"), f"{path}.metrics.samples") != samples:
        raise ValueError(f"{path}.metrics.samples mismatch")
    if _integer(metrics.get("contexts"), f"{path}.metrics.contexts") != contexts:
        raise ValueError(f"{path}.metrics.contexts mismatch")
    _object(result.get("effect"), f"{path}.effect")
    return result


def _verify_optional_interventions(
    directory: Path,
    *,
    checkpoint_hash: str,
    fisher_path: Path,
    bases: dict[str, object],
    test_metrics: object,
    samples: int,
    contexts: int,
    sequence_length: int,
) -> dict[str, object]:
    report_path = directory / "intervention_report.json"
    markdown_path = directory / "intervention_report.md"
    csv_path = directory / "intervention_results.csv"
    bundle = (report_path, markdown_path, csv_path)
    present = [path.is_file() for path in bundle]
    if not any(present):
        return {"present": False}
    if not all(present):
        missing = [
            path.name for path, exists in zip(bundle, present, strict=True) if not exists
        ]
        raise FileNotFoundError(f"incomplete intervention artifacts: {missing}")

    report = _object(
        json.loads(report_path.read_text()),
        "intervention report",
    )
    _assert_finite_tree(report, "intervention report")
    if _integer(report.get("format_version"), "intervention format") != 3:
        raise ValueError("unsupported intervention report format")
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("intervention report checkpoint hash mismatch")
    if report.get("fisher_artifact") != fisher_path.name:
        raise ValueError("intervention report Fisher artifact mismatch")
    if report.get("evaluation_split") != "test":
        raise ValueError("intervention report is not held out on test")
    _assert_metrics_match(
        report.get("baseline_test_metrics"),
        asdict(test_metrics),  # type: ignore[arg-type]
        path="intervention baseline test metrics",
    )

    config = _object(report.get("config"), "intervention config")
    boundary_values = _array(config.get("boundaries"), "intervention boundaries")
    if not boundary_values or not all(
        isinstance(boundary, str) and boundary for boundary in boundary_values
    ):
        raise ValueError("intervention boundaries are invalid")
    boundaries = tuple(str(boundary) for boundary in boundary_values)
    if len(boundaries) != len(set(boundaries)):
        raise ValueError("intervention boundaries contain duplicates")
    roles = _object(config.get("boundary_roles"), "intervention boundary roles")
    if set(roles) != set(boundaries):
        raise ValueError("intervention boundary roles mismatch")
    for boundary in boundaries:
        if boundary not in bases:
            raise ValueError(f"intervention basis is missing: {boundary}")
        basis = bases[boundary]
        position_means = basis.position_means  # type: ignore[attr-defined]
        if (
            position_means is None
            or position_means.shape
            != (sequence_length, basis.width)  # type: ignore[attr-defined]
            or not torch.isfinite(position_means).all()
        ):
            raise ValueError(
                f"intervention position means are invalid: {boundary}"
            )

    counts = tuple(
        _integer(value, "intervention necessity mode count")
        for value in _array(
            config.get("necessity_mode_counts"),
            "intervention necessity mode counts",
        )
    )
    fractions = tuple(
        _number(value, "intervention suppression fraction")
        for value in _array(
            config.get("suppression_fractions"),
            "intervention suppression fractions",
        )
    )
    random_replicates = _integer(
        config.get("random_replicates"),
        "intervention random replicates",
    )
    position_mode_count = _integer(
        config.get("position_mode_count"),
        "intervention position mode count",
    )
    if (
        not counts
        or any(count <= 0 for count in counts)
        or not fractions
        or any(not 0.0 <= fraction <= 1.0 for fraction in fractions)
        or random_replicates <= 0
        or position_mode_count <= 0
        or config.get("centering") != "validation_fisher_position_means"
    ):
        raise ValueError("intervention config is invalid")

    single = _object(report.get("single_mode"), "single-mode results")
    single_analysis = _object(
        report.get("single_mode_analysis"),
        "single-mode analysis",
    )
    if set(single) != set(boundaries) or set(single_analysis) != set(boundaries):
        raise ValueError("single-mode boundaries mismatch")
    single_count = 0
    all_results: list[object] = []
    for boundary in boundaries:
        results = _array(single[boundary], f"single-mode {boundary}")
        if len(results) != bases[boundary].width:  # type: ignore[attr-defined]
            raise ValueError(f"single-mode result count mismatch: {boundary}")
        single_count += len(results)
        all_results.extend(results)

    group = _array(report.get("group_sweep"), "group sweep")
    expected_group = (
        len(boundaries)
        * len(counts)
        * len(fractions)
        * (2 + random_replicates)
    )
    if len(group) != expected_group:
        raise ValueError("group sweep result count mismatch")
    analysis = _array(report.get("group_analysis"), "group analysis")
    expected_analysis = len(boundaries) * len(counts) * len(fractions)
    if len(analysis) != expected_analysis:
        raise ValueError("group analysis result count mismatch")

    sufficiency = _array(report.get("sufficiency"), "sufficiency results")
    expected_sufficiency = sum(
        2
        * len(
            {
                1,
                2,
                4,
                8,
                basis.modes_for_fraction(0.90),
                basis.modes_for_fraction(0.95),
                basis.modes_for_fraction(0.99),
                basis.width,
            }
        )
        for boundary, basis in bases.items()
        if boundary in boundaries
    )
    if len(sufficiency) != expected_sufficiency:
        raise ValueError("sufficiency result count mismatch")
    position_scan = _array(report.get("position_scan"), "position scan")
    if len(position_scan) != len(boundaries) * sequence_length:
        raise ValueError("position-scan result count mismatch")
    all_results.extend(group)
    all_results.extend(sufficiency)
    all_results.extend(position_scan)
    for index, result in enumerate(all_results):
        _validate_result(
            result,
            bases=bases,
            samples=samples,
            contexts=contexts,
            path=f"intervention result {index}",
        )

    primary = _object(
        report.get("primary_comparison"),
        "primary comparison",
    )
    primary_boundary = primary.get("boundary")
    if primary_boundary not in boundaries:
        raise ValueError("primary comparison boundary mismatch")
    primary_count = _integer(primary.get("mode_count"), "primary mode count")
    primary_fraction = _number(
        primary.get("suppression_fraction"),
        "primary suppression fraction",
    )
    if primary_count not in counts or primary_fraction not in fractions:
        raise ValueError("primary comparison is outside the configured sweep")
    for name in ("top", "bottom"):
        result = _validate_result(
            primary.get(name),
            bases=bases,
            samples=samples,
            contexts=contexts,
            path=f"primary {name}",
        )
        if (
            result.get("boundary") != primary_boundary
            or _integer(result.get("mode_count"), f"primary {name} mode count")
            != primary_count
        ):
            raise ValueError(f"primary {name} structure mismatch")
    random_control = _object(
        primary.get("random_control"),
        "primary random control",
    )
    if (
        random_control.get("boundary") != primary_boundary
        or _integer(
            random_control.get("random_replicates"),
            "primary random-control count",
        )
        != random_replicates
    ):
        raise ValueError("primary random-control structure mismatch")
    _validate_bootstrap(
        primary.get("paired_context_top_minus_bottom_bootstrap"),
        contexts=contexts,
        path="primary bootstrap",
    )

    energy = _object(
        primary.get("energy_matched_controls"),
        "energy-matched controls",
    )
    energy_bottom = _validate_result(
        energy.get("bottom"),
        bases=bases,
        samples=samples,
        contexts=contexts,
        path="energy-matched bottom",
    )
    if (
        energy_bottom.get("boundary") != primary_boundary
        or _integer(
            energy_bottom.get("mode_count"),
            "energy-matched bottom mode count",
        )
        != primary_count
    ):
        raise ValueError("energy-matched bottom structure mismatch")
    bottom_calibration_rms = _number(
        energy_bottom.get("calibration_activation_delta_rms"),
        "energy-matched bottom calibration RMS",
    )
    energy_random = _array(
        energy.get("random_results"),
        "energy-matched random results",
    )
    if len(energy_random) != random_replicates:
        raise ValueError("energy-matched random result count mismatch")
    random_calibration_rms: list[float] = []
    random_test_rms: list[float] = []
    for index, result in enumerate(energy_random):
        item = _validate_result(
            result,
            bases=bases,
            samples=samples,
            contexts=contexts,
            path=f"energy-matched random result {index}",
        )
        if (
            item.get("boundary") != primary_boundary
            or item.get("control") != "energy_matched_random"
            or _integer(
                item.get("mode_count"),
                "energy-matched random mode count",
            )
            != primary_count
        ):
            raise ValueError("energy-matched random structure mismatch")
        random_calibration_rms.append(
            _number(
                item.get("calibration_activation_delta_rms"),
                f"energy-matched random calibration RMS {index}",
            )
        )
        random_test_rms.append(
            _number(
                item.get("activation_delta_rms"),
                f"energy-matched random test RMS {index}",
            )
        )
    energy_summary = _object(
        energy.get("random_summary"),
        "energy-matched random summary",
    )
    if _integer(
        energy_summary.get("random_replicates"),
        "energy-matched random summary count",
    ) != random_replicates:
        raise ValueError("energy-matched random summary count mismatch")
    if energy_summary.get("calibration_split") != "validation_fisher":
        raise ValueError("energy matching was not calibrated on validation_fisher")
    target_calibration_rms = _number(
        energy_summary.get("target_calibration_activation_delta_rms"),
        "energy-matched target calibration RMS",
    )
    if not math.isclose(
        bottom_calibration_rms,
        target_calibration_rms,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ) or any(
        not math.isclose(
            value,
            target_calibration_rms,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        for value in random_calibration_rms
    ):
        raise ValueError("energy-matched calibration RMS does not match target")
    ranges = {
        "actual_calibration_activation_delta_rms_range": (
            min(random_calibration_rms),
            max(random_calibration_rms),
        ),
        "actual_test_activation_delta_rms_range": (
            min(random_test_rms),
            max(random_test_rms),
        ),
    }
    for field, expected_range in ranges.items():
        interval = _array(energy_summary.get(field), f"energy summary {field}")
        if len(interval) != 2:
            raise ValueError(f"energy summary {field} must have two values")
        actual_range = (
            _number(interval[0], f"energy summary {field}[0]"),
            _number(interval[1], f"energy summary {field}[1]"),
        )
        if actual_range[0] > actual_range[1] or any(
            not math.isclose(
                actual,
                expected,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
            for actual, expected in zip(
                actual_range, expected_range, strict=True
            )
        ):
            raise ValueError(f"energy summary {field} mismatch")
    delta_interval = _array(
        energy_summary.get("delta_hard_nll_95_interval"),
        "energy summary delta_hard_nll_95_interval",
    )
    if len(delta_interval) != 2:
        raise ValueError(
            "energy summary delta_hard_nll_95_interval must have two values"
        )
    _validate_bootstrap(
        energy.get("paired_context_top_minus_bottom_bootstrap"),
        contexts=contexts,
        path="energy-matched bootstrap",
    )

    if not markdown_path.read_text().strip():
        raise ValueError("intervention Markdown report is empty")
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if (
            reader.fieldnames is None
            or "calibration_activation_delta_rms" not in reader.fieldnames
        ):
            raise ValueError("intervention CSV lacks calibration RMS column")
        csv_rows = list(reader)
    expected_csv_rows = (
        single_count
        + len(group)
        + len(sufficiency)
        + len(position_scan)
        + 1
        + len(energy_random)
    )
    if len(csv_rows) != expected_csv_rows:
        raise ValueError("intervention CSV row count mismatch")
    for index, row in enumerate(csv_rows):
        calibration_value = row["calibration_activation_delta_rms"]
        if row.get("experiment") == "energy_matched_primary":
            _number(
                float(calibration_value),
                f"intervention CSV calibration RMS row {index}",
            )
        elif calibration_value not in ("", None):
            raise ValueError(
                "intervention CSV has calibration RMS outside matched rows"
            )

    return {
        "present": True,
        "format_version": report["format_version"],
        "boundary_count": len(boundaries),
        "single_mode_count": single_count,
        "group_sweep_count": len(group),
        "group_analysis_count": len(analysis),
        "sufficiency_count": len(sufficiency),
        "position_scan_count": len(position_scan),
        "energy_random_count": len(energy_random),
        "csv_row_count": len(csv_rows),
        "status": "verified",
    }


def _verify_optional_modal_executor(
    directory: Path,
    *,
    artifact_layer_index: int,
    checkpoint_hash: str,
    fisher_path: Path,
    bases: dict[str, FisherModeBasis],
    model: ToyTransformer,
    model_config: TransformerConfig,
    splits: AssociativeRecallSplits,
    manifest: dict[str, object],
    validation_metrics: AssociativeRecallMetrics,
    test_metrics: AssociativeRecallMetrics,
    sequence_length: int,
) -> dict[str, object]:
    artifact_paths = modal_executor_artifact_paths(
        directory,
        artifact_layer_index,
    )
    executor_path = artifact_paths.executor
    report_path = artifact_paths.report_json
    markdown_path = artifact_paths.report_markdown
    bundle = (executor_path, report_path, markdown_path)
    present = [path.is_file() for path in bundle]
    if not any(present):
        return {"present": False}
    if not all(present):
        missing = [
            path.name
            for path, exists in zip(bundle, present, strict=True)
            if not exists
        ]
        raise FileNotFoundError(f"incomplete modal executor artifacts: {missing}")

    report = _object(
        json.loads(report_path.read_text()),
        "modal executor report",
    )
    _assert_finite_tree(report, "modal executor report")
    expected_report_fields = {
        "format_version",
        "checkpoint_sha256",
        "fisher_sha256",
        "modal_executor_sha256",
        "layer_index",
        "teacher_state_sha256_before",
        "teacher_state_sha256_after",
        "teacher_was_frozen",
        "training_distribution",
        "training_contract",
        "target",
        "robustification_used",
        "compensation_target_used",
        "input_activation",
        "output_activation",
        "fit_split",
        "selection_split",
        "evaluation_split",
        "test_used_for_fit_or_selection",
        "fit_protocol",
        "baseline",
        "bottleneck",
        "affine_graph",
        "nonlinear_candidates",
        "nonlinear_graph",
        "artifacts",
        "elapsed_seconds",
        "scientific_status",
    }
    if set(report) != expected_report_fields:
        raise ValueError("modal executor report fields mismatch")
    if _integer(report.get("format_version"), "modal executor format") != 1:
        raise ValueError("unsupported modal executor report format")
    fisher_hash = _sha256(fisher_path)
    executor_hash = _sha256(executor_path)
    expected_hashes = (
        ("checkpoint_sha256", checkpoint_hash),
        ("fisher_sha256", fisher_hash),
        ("modal_executor_sha256", executor_hash),
    )
    for name, expected in expected_hashes:
        if report.get(name) != expected:
            raise ValueError(f"modal executor report {name} mismatch")

    artifacts = _object(
        report.get("artifacts"),
        "modal executor report artifacts",
    )
    expected_artifacts = {
        "executor": executor_path.name,
        "checkpoint": "checkpoint.pt",
        "fisher": fisher_path.name,
    }
    if artifacts != expected_artifacts:
        raise ValueError("modal executor report artifact names mismatch")
    if (
        report.get("scientific_status")
        != "exploratory_single_checkpoint_test_previously_inspected"
    ):
        raise ValueError("modal executor scientific status mismatch")
    if _number(
        report.get("elapsed_seconds"),
        "modal executor report elapsed_seconds",
    ) < 0:
        raise ValueError("modal executor elapsed_seconds is negative")

    executor, config, metadata = load_position_modal_executor(executor_path)
    _assert_finite_tree(metadata, "modal executor metadata")
    if not isinstance(executor, PositionConditionedModalGraphExecutor):
        raise ValueError("modal executor artifact did not load a graph executor")
    if not isinstance(executor.graph, CausalModalMLPGraph):
        raise ValueError("modal executor graph is not the causal nonlinear graph")
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("modal executor checkpoint hash mismatch")
    if metadata.get("fisher_sha256") != fisher_hash:
        raise ValueError("modal executor Fisher hash mismatch")
    teacher_state_hash = _module_state_sha256(model)
    if (
        report.get("teacher_state_sha256_before")
        != teacher_state_hash
        or report.get("teacher_state_sha256_after")
        != teacher_state_hash
        or metadata.get("teacher_state_sha256")
        != teacher_state_hash
    ):
        raise ValueError("modal executor teacher state hash mismatch")
    frozen_contract = {
        "teacher_was_frozen": True,
        "training_distribution": (
            "clean_frozen_teacher_boundary_pairs"
        ),
        "training_contract": "same_forward_input_output",
        "target": "frozen_teacher_layer_output_for_exact_input",
        "robustification_used": False,
        "compensation_target_used": False,
    }
    for name, expected in frozen_contract.items():
        if report.get(name) != expected or metadata.get(name) != expected:
            raise ValueError(
                f"modal executor frozen contract {name} mismatch"
            )
    fit_protocol, fit_config = _validate_modal_fit_protocol(
        report.get("fit_protocol"),
        metadata.get("fit_protocol"),
    )
    minimum_scale = _number(
        fit_protocol.get("minimum_scale"),
        "modal executor fit protocol.minimum_scale",
    )

    split_map = {
        "train": splits.train,
        "validation_fisher": splits.validation,
        "test": splits.test,
    }
    split_hashes: dict[str, str] = {}
    for name, split in split_map.items():
        section = _object(manifest.get(name), f"split manifest {name}")
        ids = [
            _integer(value, f"split manifest {name}.context_ids")
            for value in _array(
                section.get("context_ids"),
                f"split manifest {name}.context_ids",
            )
        ]
        if ids != split.context_ids.tolist():
            raise ValueError(f"split manifest context IDs mismatch: {name}")
        split_hash = _tensor_sha256(split.context_ids)
        if section.get("context_ids_sha256") != split_hash:
            raise ValueError(f"split manifest context hash mismatch: {name}")
        split_hashes[name] = split_hash

    provenance = (
        ("fit_split", "train"),
        ("selection_split", "validation_fisher"),
    )
    for name, expected in provenance:
        if report.get(name) != expected or metadata.get(name) != expected:
            raise ValueError(f"modal executor {name} provenance mismatch")
    if report.get("evaluation_split") != "test":
        raise ValueError("modal executor evaluation is not held out on test")
    if (
        report.get("test_used_for_fit_or_selection") is not False
        or metadata.get("test_used_for_fit_or_selection") is not False
    ):
        raise ValueError("modal executor used test for fit or selection")
    if metadata.get("fit_context_ids_sha256") != split_hashes["train"]:
        raise ValueError("modal executor fit context hash mismatch")
    if (
        metadata.get("selection_context_ids_sha256")
        != split_hashes["validation_fisher"]
    ):
        raise ValueError("modal executor selection context hash mismatch")

    layer_index = _integer(report.get("layer_index"), "modal layer index")
    if not 0 <= layer_index < len(model.layers):
        raise ValueError("modal executor layer index is outside the model")
    if layer_index != artifact_layer_index:
        raise ValueError(
            "modal executor layer index does not match artifact name"
        )
    if _integer(metadata.get("layer_index"), "modal metadata layer index") != (
        layer_index
    ):
        raise ValueError("modal executor layer index metadata mismatch")
    expected_input = (
        "layer.0.input"
        if layer_index == 0
        else f"layer.{layer_index - 1}.output"
    )
    expected_output = f"layer.{layer_index}.output"
    if (
        config.input_activation != expected_input
        or config.output_activation != expected_output
        or report.get("input_activation") != expected_input
        or report.get("output_activation") != expected_output
    ):
        raise ValueError("modal executor replacement boundaries mismatch")
    if expected_input not in bases or expected_output not in bases:
        raise ValueError("modal executor Fisher bases are missing")
    if config.sequence_length != sequence_length:
        raise ValueError("modal executor sequence length mismatch")

    input_basis = bases[expected_input]
    output_basis = bases[expected_output]
    if (
        not 1 <= config.input_modes <= input_basis.width
        or not 1 <= config.output_modes <= output_basis.width
    ):
        raise ValueError("modal executor mode count is outside its Fisher basis")
    if config.routing_width != executor.graph.hidden_modes:
        raise ValueError("modal executor routing width mismatch")
    if config.sequence_length != executor.graph.sequence_length:
        raise ValueError("modal executor graph sequence length mismatch")
    if (
        executor.input_projection.activation_name != expected_input
        or executor.output_projection.activation_name != expected_output
    ):
        raise ValueError("modal executor projection activation names mismatch")
    if (
        executor.input_projection.width != model_config.d_model
        or executor.output_projection.width != model_config.d_model
        or executor.input_projection.modes != config.input_modes
        or executor.output_projection.modes != config.output_modes
    ):
        raise ValueError("modal executor projection dimensions mismatch")

    for name, projection, basis, modes in (
        (
            "input",
            executor.input_projection,
            input_basis,
            config.input_modes,
        ),
        (
            "output",
            executor.output_projection,
            output_basis,
            config.output_modes,
        ),
    ):
        if basis.position_means is None:
            raise ValueError(f"modal executor {name} basis lacks position means")
        torch.testing.assert_close(
            projection.position_mean,
            basis.position_means.to(projection.position_mean),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            projection.vectors,
            basis.vectors[:, :modes].to(projection.vectors),
            rtol=0,
            atol=0,
        )

    state = executor.state_dict()
    if not state:
        raise ValueError("modal executor state is empty")
    for name, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite modal executor state: {name}")
    for name, scale in (
        ("input_scale", executor.graph.input_scale),
        ("output_scale", executor.graph.output_scale),
    ):
        if not torch.isfinite(scale).all() or (scale < minimum_scale).any():
            raise ValueError(
                f"modal executor {name} violates the fitted scale floor"
            )

    if hasattr(executor, "inner") or any(
        isinstance(module, TransformerBlock) for module in executor.modules()
    ):
        raise ValueError("modal graph executor retains a transformer block")
    if any(
        not name.startswith("graph.")
        for name, _ in executor.named_parameters()
    ):
        raise ValueError("modal executor has parameters outside the graph")
    if (
        len(executor.graph.input_layers) != sequence_length
        or len(executor.graph.output_layers) != sequence_length
    ):
        raise ValueError("modal graph position-node count mismatch")
    for position, layer in enumerate(executor.graph.input_layers):
        if (
            layer.in_features != (position + 1) * config.input_modes
            or layer.out_features != config.routing_width
        ):
            raise ValueError(
                f"modal graph input node {position} is not causally shaped"
            )
    for position, layer in enumerate(executor.graph.output_layers):
        if (
            layer.in_features != config.routing_width
            or layer.out_features != config.output_modes
        ):
            raise ValueError(
                f"modal graph output node {position} has the wrong shape"
            )

    if sequence_length > 1:
        dtype = executor.input_projection.position_mean.dtype
        probe = torch.zeros(
            1,
            sequence_length,
            model_config.d_model,
            dtype=dtype,
        )
        changed = probe.clone()
        changed[:, -1] = 1
        attention_mask = torch.ones(
            1,
            sequence_length,
            dtype=torch.bool,
        )
        executor.eval()
        with torch.no_grad():
            original_output = executor(
                probe,
                attention_mask=attention_mask,
                trace=None,
                prefix=f"layer.{layer_index}",
            )
            changed_output = executor(
                changed,
                attention_mask=attention_mask,
                trace=None,
                prefix=f"layer.{layer_index}",
            )
        torch.testing.assert_close(
            original_output[:, :-1],
            changed_output[:, :-1],
            rtol=0,
            atol=0,
        )

    validation_template = validation_metrics
    test_template = test_metrics
    baseline = _object(report.get("baseline"), "modal baseline")
    _validate_metrics_structure(
        baseline.get("validation_metrics"),
        template=validation_template,
        samples=splits.validation.samples,
        contexts=splits.validation.contexts,
        path="modal baseline validation metrics",
    )
    _validate_metrics_structure(
        baseline.get("test_metrics"),
        template=test_template,
        samples=splits.test.samples,
        contexts=splits.test.contexts,
        path="modal baseline test metrics",
    )
    _assert_metrics_match(
        baseline.get("validation_metrics"),
        asdict(validation_metrics),
        path="modal baseline validation metrics",
    )
    _assert_metrics_match(
        baseline.get("test_metrics"),
        asdict(test_metrics),
        path="modal baseline test metrics",
    )

    bottleneck = _object(report.get("bottleneck"), "modal bottleneck")
    for split_name, split, template in (
        ("validation", splits.validation, validation_template),
        ("test", splits.test, test_template),
    ):
        _validate_metrics_structure(
            bottleneck.get(f"{split_name}_metrics"),
            template=template,
            samples=split.samples,
            contexts=split.contexts,
            path=f"modal bottleneck {split_name} metrics",
        )

    affine = _object(report.get("affine_graph"), "modal affine graph")
    affine_fit = _object(affine.get("fit"), "modal affine graph fit")
    if _integer(
        affine_fit.get("samples"),
        "modal affine graph fit.samples",
    ) != splits.train.samples:
        raise ValueError("modal affine fit sample count mismatch")
    for name in ("train_rmse", "ridge"):
        if _number(
            affine_fit.get(name),
            f"modal affine graph fit.{name}",
        ) < 0:
            raise ValueError(f"modal affine graph fit.{name} is negative")
    _number(
        affine_fit.get("train_r_squared"),
        "modal affine graph fit.train_r_squared",
    )
    if _integer(
        affine_fit.get("learned_parameters"),
        "modal affine graph fit.learned_parameters",
    ) <= 0:
        raise ValueError("modal affine learned parameter count is invalid")
    for split_name, split, template in (
        ("validation", splits.validation, validation_template),
        ("test", splits.test, test_template),
    ):
        _validate_metrics_structure(
            affine.get(f"{split_name}_metrics"),
            template=template,
            samples=split.samples,
            contexts=split.contexts,
            path=f"modal affine {split_name} metrics",
        )
    for section_name, section in (
        ("affine", affine),
        (
            "nonlinear",
            _object(report.get("nonlinear_graph"), "modal nonlinear graph"),
        ),
    ):
        activation_fit = _object(
            section.get("validation_activation_fit"),
            f"modal {section_name} activation fit",
        )
        _number(
            activation_fit.get("activation_r_squared"),
            f"modal {section_name} activation fit.activation_r_squared",
        )
        if _number(
            activation_fit.get("activation_rmse"),
            f"modal {section_name} activation fit.activation_rmse",
        ) < 0:
            raise ValueError(
                f"modal {section_name} activation RMSE is negative"
            )

    graph_edges = executor.graph.edge_count
    learned_parameters = sum(
        parameter.numel() for parameter in executor.graph.parameters()
    )
    candidates = _array(
        report.get("nonlinear_candidates"),
        "modal nonlinear candidates",
    )
    if not candidates:
        raise ValueError("modal nonlinear candidates cannot be empty")
    candidate_by_width: dict[int, dict[str, object]] = {}
    total_history_points = 0
    gate = _object(metadata.get("validation_gate"), "modal validation gate")
    minimum_answer = _number(
        gate.get("minimum_answer_accuracy"),
        "modal validation gate.minimum_answer_accuracy",
    )
    minimum_paired = _number(
        gate.get("minimum_paired_accuracy"),
        "modal validation gate.minimum_paired_accuracy",
    )
    maximum_nll_increase = _number(
        gate.get("maximum_nll_increase"),
        "modal validation gate.maximum_nll_increase",
    )
    if (
        not 0 <= minimum_answer <= 1
        or not 0 <= minimum_paired <= 1
        or maximum_nll_increase < 0
    ):
        raise ValueError("modal validation gate is invalid")
    baseline_validation_nll = validation_metrics.hard_nll

    for index, value in enumerate(candidates):
        path = f"modal nonlinear candidate {index}"
        candidate = _object(value, path)
        routing_width = _integer(
            candidate.get("routing_width"),
            f"{path}.routing_width",
        )
        if routing_width <= 0 or routing_width in candidate_by_width:
            raise ValueError("modal candidate routing widths are invalid")
        candidate_fit_config = _object(
            candidate.get("fit_config"),
            f"{path}.fit_config",
        )
        if _json_normalized(candidate_fit_config) != _json_normalized(
            fit_config
        ):
            raise ValueError(f"{path}.fit_config does not match fit protocol")
        expected_edges = (
            sequence_length
            * (sequence_length + 1)
            // 2
            * config.input_modes
            * routing_width
            + sequence_length
            * routing_width
            * config.output_modes
        )
        expected_parameters = (
            expected_edges
            + sequence_length * routing_width
            + sequence_length * config.output_modes
        )
        _, history_count = _validate_modal_fit(
            candidate.get("fit"),
            path=f"{path}.fit",
            expected_parameters=expected_parameters,
            expected_edges=expected_edges,
        )
        _validate_modal_history_schedule(
            candidate.get("fit"),
            fit_config,
            path=f"{path}.fit",
        )
        total_history_points += history_count
        candidate_metrics = _validate_metrics_structure(
            candidate.get("validation_metrics"),
            template=validation_template,
            samples=splits.validation.samples,
            contexts=splits.validation.contexts,
            path=f"{path}.validation_metrics",
        )
        passed = (
            _number(
                candidate_metrics.get("answer_accuracy"),
                f"{path}.answer_accuracy",
            )
            >= minimum_answer
            and _number(
                candidate_metrics.get("paired_context_accuracy"),
                f"{path}.paired_context_accuracy",
            )
            >= minimum_paired
            and _number(
                candidate_metrics.get("hard_nll"),
                f"{path}.hard_nll",
            )
            <= baseline_validation_nll + maximum_nll_increase
        )
        if candidate.get("validation_gate_passed") is not passed:
            raise ValueError(f"{path}.validation_gate_passed mismatch")
        candidate_by_width[routing_width] = candidate

    passing_widths = [
        width
        for width, candidate in candidate_by_width.items()
        if candidate["validation_gate_passed"]
    ]
    if passing_widths:
        expected_selected_width = min(passing_widths)
    else:
        expected_selected_width = min(
            candidate_by_width,
            key=lambda width: _number(
                _object(
                    candidate_by_width[width]["validation_metrics"],
                    "modal candidate validation metrics",
                ).get("hard_nll"),
                "modal candidate validation hard_nll",
            ),
        )
    if config.routing_width != expected_selected_width:
        raise ValueError("modal executor candidate selection mismatch")
    selected_candidate = candidate_by_width[config.routing_width]
    if _json_normalized(metadata.get("selected_candidate")) != selected_candidate:
        raise ValueError("modal executor selected-candidate metadata mismatch")

    nonlinear = _object(report.get("nonlinear_graph"), "modal nonlinear graph")
    if nonlinear.get("config") != asdict(config):
        raise ValueError("modal nonlinear graph config mismatch")
    if _json_normalized(nonlinear.get("fit_config")) != _json_normalized(
        fit_config
    ):
        raise ValueError("modal nonlinear graph fit config mismatch")
    if _json_normalized(nonlinear.get("fit")) != selected_candidate.get("fit"):
        raise ValueError("modal nonlinear graph fit mismatch")
    if (
        nonlinear.get("validation_metrics")
        != selected_candidate.get("validation_metrics")
    ):
        raise ValueError("modal nonlinear validation metrics mismatch")
    if (
        nonlinear.get("validation_gate_passed")
        is not selected_candidate.get("validation_gate_passed")
    ):
        raise ValueError("modal nonlinear validation gate flag mismatch")
    _, selected_history_count = _validate_modal_fit(
        nonlinear.get("fit"),
        path="modal nonlinear graph fit",
        expected_parameters=learned_parameters,
        expected_edges=graph_edges,
    )
    _validate_metrics_structure(
        nonlinear.get("validation_metrics"),
        template=validation_template,
        samples=splits.validation.samples,
        contexts=splits.validation.contexts,
        path="modal nonlinear validation metrics",
    )
    _validate_metrics_structure(
        nonlinear.get("test_metrics"),
        template=test_template,
        samples=splits.test.samples,
        contexts=splits.test.contexts,
        path="modal nonlinear test metrics",
    )

    size = _object(nonlinear.get("size"), "modal nonlinear graph size")
    original_parameters = sum(
        parameter.numel() for parameter in model.layers[layer_index].parameters()
    )
    expected_multiplies = (
        sequence_length
        * model_config.d_model
        * (config.input_modes + config.output_modes)
        + graph_edges
    )
    original_multiplies = (
        4
        * sequence_length
        * model_config.d_model
        * model_config.d_model
        + 2
        * sequence_length
        * sequence_length
        * model_config.d_model
        + 2
        * sequence_length
        * model_config.d_model
        * model_config.d_ff
    )
    expected_counts = (
        ("learned_parameters", learned_parameters),
        ("original_block_parameters", original_parameters),
        ("graph_edges", graph_edges),
        ("estimated_multiplies_per_sequence", expected_multiplies),
        (
            "original_block_estimated_multiplies_per_sequence",
            original_multiplies,
        ),
    )
    for name, expected in expected_counts:
        if _integer(size.get(name), f"modal size.{name}") != expected:
            raise ValueError(f"modal size.{name} mismatch")
    expected_ratio = expected_multiplies / original_multiplies
    if not math.isclose(
        _number(
            size.get("estimated_multiply_ratio"),
            "modal size.estimated_multiply_ratio",
        ),
        expected_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("modal size.estimated_multiply_ratio mismatch")

    original_layer = model.layers[layer_index]
    model.replace_layer(layer_index, executor)
    try:
        executor_validation_metrics = evaluate_associative_recall(
            model,
            splits.validation,
        )
        executor_test_metrics = evaluate_associative_recall(
            model,
            splits.test,
        )
    finally:
        model.replace_layer(layer_index, original_layer)
    _assert_metrics_match(
        nonlinear.get("validation_metrics"),
        asdict(executor_validation_metrics),
        path="modal recomputed validation metrics",
    )
    _assert_metrics_match(
        nonlinear.get("test_metrics"),
        asdict(executor_test_metrics),
        path="modal recomputed test metrics",
    )

    if not markdown_path.read_text().strip():
        raise ValueError("modal executor Markdown report is empty")
    return {
        "present": True,
        "format_version": report["format_version"],
        "layer_index": layer_index,
        "input_modes": config.input_modes,
        "output_modes": config.output_modes,
        "routing_width": config.routing_width,
        "fit_steps": fit_config["steps"],
        "fit_seed": fit_config["seed"],
        "coordinate_normalization": fit_protocol[
            "coordinate_normalization"
        ],
        "minimum_scale": minimum_scale,
        "initialization_count": fit_protocol["initialization_count"],
        "candidate_count": len(candidates),
        "candidate_history_point_count": total_history_points,
        "selected_history_point_count": selected_history_count,
        "state_tensor_count": len(state),
        "learned_parameters": learned_parameters,
        "graph_edges": graph_edges,
        "test_accuracy": executor_test_metrics.answer_accuracy,
        "test_paired_accuracy": (
            executor_test_metrics.paired_context_accuracy
        ),
        "test_nll": executor_test_metrics.hard_nll,
        "status": "verified",
    }


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _assert_numeric_tree_match(
    actual: object,
    expected: object,
    *,
    path: str,
) -> None:
    if isinstance(expected, dict):
        actual_object = _object(actual, path)
        if set(actual_object) != set(expected):
            raise ValueError(f"{path} fields mismatch")
        for name, value in expected.items():
            _assert_numeric_tree_match(
                actual_object[name],
                value,
                path=f"{path}.{name}",
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            raise ValueError(f"{path} must be an array")
        if len(actual) != len(expected):
            raise ValueError(f"{path} length mismatch")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_numeric_tree_match(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{path} mismatch")
        return
    if isinstance(expected, int):
        if _integer(actual, path) != expected:
            raise ValueError(f"{path} mismatch")
        return
    if isinstance(expected, float):
        if not math.isclose(
            _number(actual, path),
            expected,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{path} mismatch")
        return
    if actual != expected:
        raise ValueError(f"{path} mismatch")


@torch.no_grad()
def _collect_completion_activations(
    model: ToyTransformer,
    split,
    names: tuple[str, ...],
    *,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {
        name: [] for name in names
    }
    was_training = model.training
    model.eval()
    try:
        for start in range(0, split.samples, batch_size):
            output = model(
                split.input_ids[start : start + batch_size],
                capture_activations=True,
                retain_activation_gradients=False,
            )
            if output.activations is None:
                raise RuntimeError(
                    "completion verification did not capture activations"
                )
            for name in names:
                captured[name].append(
                    output.activations[name].detach().cpu()
                )
    finally:
        model.train(was_training)
    return {
        name: torch.cat(values, dim=0)
        for name, values in captured.items()
    }


@torch.no_grad()
def _completion_replacement_logits(
    model: ToyTransformer,
    *,
    layer_index: int,
    executor,
    split,
) -> torch.Tensor:
    original = model.layers[layer_index]
    model.replace_layer(layer_index, executor)
    try:
        return associative_recall_answer_logits(model, split)
    finally:
        model.replace_layer(layer_index, original)


def _completion_behavior(
    split,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> dict[str, object]:
    metrics = associative_recall_metrics_from_logits(split, logits)
    teacher_log_probabilities = teacher_logits.log_softmax(dim=-1)
    teacher_probabilities = teacher_log_probabilities.exp()
    candidate_log_probabilities = logits.log_softmax(dim=-1)
    kl = (
        teacher_probabilities
        * (teacher_log_probabilities - candidate_log_probabilities)
    ).sum(dim=-1).clamp_min(0)
    return {
        "metrics": asdict(metrics),
        "teacher_to_system_answer_kl": kl.mean().item(),
        "maximum_answer_kl": kl.max().item(),
    }


def _completion_r_squared(
    actual: torch.Tensor,
    predicted: torch.Tensor,
) -> float:
    residual = (actual - predicted).square().sum()
    centered = (
        actual - actual.mean(dim=0, keepdim=True)
    ).square().sum()
    return (
        1.0 - (residual / centered).item()
        if centered > 0
        else 1.0
    )


@torch.no_grad()
def _completion_local_metrics(
    completion: PositionConditionedModalCompletion,
    mean_control: PositionConditionedModalCompletion,
    activations: torch.Tensor,
    basis: FisherModeBasis,
    *,
    train_tail_scale: torch.Tensor,
) -> dict[str, float | int]:
    full = basis.project(
        activations.to(torch.float64),
        modes=basis.width,
        centering="position",
    )
    kept = full[..., : completion.kept_modes]
    actual_tail = full[..., completion.kept_modes :]
    predicted_tail = completion.graph(
        kept.to(
            dtype=completion.graph.weight.dtype,
            device=completion.graph.weight.device,
        )
    ).to(torch.float64)
    mean_tail = mean_control.graph(
        kept.to(
            dtype=mean_control.graph.weight.dtype,
            device=mean_control.graph.weight.device,
        )
    ).to(torch.float64)
    zero_mse = actual_tail.square().mean()
    mean_mse = (actual_tail - mean_tail).square().mean()
    completion_mse = (actual_tail - predicted_tail).square().mean()
    reconstructed = completion.decode(
        torch.cat(
            (kept.to(predicted_tail), predicted_tail),
            dim=-1,
        ).to(completion.full_projection.vectors)
    ).to(torch.float64)
    activation_residual = activations.to(torch.float64) - reconstructed
    position_r_squared: list[float] = []
    constant_positions = 0
    for position in range(actual_tail.shape[1]):
        actual = actual_tail[:, position]
        predicted = predicted_tail[:, position]
        centered_sum = (
            actual - actual.mean(dim=0, keepdim=True)
        ).square().sum()
        if centered_sum <= torch.finfo(actual.dtype).eps:
            constant_positions += 1
            continue
        position_r_squared.append(
            1.0
            - (
                (actual - predicted).square().sum()
                / centered_sum
            ).item()
        )
    centered_activations = (
        activations.to(torch.float64)
        - activations.to(torch.float64).mean(dim=0, keepdim=True)
    )
    return {
        "samples": activations.shape[0],
        "tail_r_squared": _completion_r_squared(
            actual_tail,
            predicted_tail,
        ),
        "tail_rmse": completion_mse.sqrt().item(),
        "tail_standardized_mse": (
            (
                (actual_tail - predicted_tail)
                / train_tail_scale.to(torch.float64)
            )
            .square()
            .mean()
            .item()
        ),
        "minimum_nonconstant_position_r_squared": (
            min(position_r_squared) if position_r_squared else 1.0
        ),
        "constant_position_count": constant_positions,
        "tail_mse_ratio_vs_zero": (
            (completion_mse / zero_mse).item()
            if zero_mse > 0
            else 0.0
        ),
        "tail_mse_ratio_vs_mean": (
            (completion_mse / mean_mse).item()
            if mean_mse > 0
            else 0.0
        ),
        "full_activation_r_squared": _completion_r_squared(
            activations.to(torch.float64),
            reconstructed,
        ),
        "full_activation_rmse": (
            activation_residual.square().mean().sqrt().item()
        ),
        "relative_centered_residual_norm": (
            activation_residual.norm() / centered_activations.norm()
        ).item(),
    }


def _completion_train_tail_scale(
    activations: torch.Tensor,
    basis: FisherModeBasis,
    *,
    kept_modes: int,
    minimum_scale: float,
) -> torch.Tensor:
    full = basis.project(
        activations.to(torch.float64),
        modes=basis.width,
        centering="position",
    )
    return full[..., kept_modes:].std(dim=0).clamp_min(
        minimum_scale
    )


@torch.no_grad()
def _completion_activation_fit(
    executor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    predictions = executor(
        inputs,
        attention_mask=torch.ones(
            inputs.shape[:2],
            dtype=torch.bool,
            device=inputs.device,
        ),
        trace=None,
        prefix=prefix,
    )
    residual = targets - predictions
    residual_sum = residual.square().sum()
    centered_sum = (
        targets - targets.mean(dim=0, keepdim=True)
    ).square().sum()
    return {
        "activation_r_squared": (
            1.0 - (residual_sum / centered_sum).item()
            if centered_sum > 0
            else 1.0
        ),
        "activation_rmse": residual.square().mean().sqrt().item(),
    }


def _completion_state_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in module.state_dict().values()
    )


def _validate_completion_bridge(
    completion: PositionConditionedModalCompletion,
    *,
    label: str,
    basis: FisherModeBasis,
    activation_name: str,
    sequence_length: int,
    width: int,
    kept_modes: int,
    graph_kind: str,
) -> None:
    config = completion.config
    expected_config = {
        "activation_name": activation_name,
        "sequence_length": sequence_length,
        "width": width,
        "kept_modes": kept_modes,
        "graph_kind": graph_kind,
    }
    if asdict(config) != expected_config:
        raise ValueError(f"{label} completion config mismatch")
    if basis.position_means is None:
        raise ValueError(f"{label} completion basis lacks position means")
    torch.testing.assert_close(
        completion.full_projection.position_mean,
        basis.position_means.to(
            completion.full_projection.position_mean
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        completion.full_projection.vectors,
        basis.vectors.to(completion.full_projection.vectors),
        rtol=0,
        atol=0,
    )
    graph = completion.graph
    tail_modes = width - kept_modes
    expected_weight_shape = (
        (kept_modes, tail_modes)
        if graph_kind == "shared_local_linear"
        else (sequence_length, kept_modes, tail_modes)
    )
    if tuple(graph.weight.shape) != expected_weight_shape:
        raise ValueError(f"{label} completion weight shape mismatch")
    if tuple(graph.bias.shape) != (sequence_length, tail_modes):
        raise ValueError(f"{label} completion bias shape mismatch")
    if graph.shared_weights != (
        graph_kind == "shared_local_linear"
    ):
        raise ValueError(f"{label} completion graph kind mismatch")
    if not all(
        name.startswith("graph.")
        for name, _ in completion.named_parameters()
    ):
        raise ValueError(
            f"{label} completion has parameters outside its graph"
        )
    if any(
        isinstance(module, TransformerBlock)
        for module in completion.modules()
    ):
        raise ValueError(
            f"{label} completion artifact contains a transformer block"
        )
    for name, tensor in completion.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(
                f"{label} completion state is nonfinite: {name}"
            )
    expected_edges = (
        sequence_length * kept_modes * tail_modes
    )
    if graph.edge_count != expected_edges:
        raise ValueError(f"{label} completion edge count mismatch")
    expected_parameters = (
        (
            kept_modes * tail_modes
            if graph.shared_weights
            else expected_edges
        )
        + sequence_length * tail_modes
    )
    if graph.learned_parameter_count != expected_parameters:
        raise ValueError(
            f"{label} completion learned parameter count mismatch"
        )

    generator = torch.Generator(device="cpu").manual_seed(
        71 if label == "input" else 73
    )
    probe = torch.randn(
        2,
        sequence_length,
        kept_modes,
        generator=generator,
        dtype=graph.weight.dtype,
    )
    with torch.no_grad():
        original = graph(probe)
    for position in range(sequence_length):
        changed = probe.clone()
        changed[:, :position] += 0.25
        changed[:, position + 1 :] -= 0.375
        with torch.no_grad():
            changed_output = graph(changed)
        torch.testing.assert_close(
            original[:, position],
            changed_output[:, position],
            rtol=0,
            atol=0,
        )
    if sequence_length > 1:
        activations = torch.randn(
            2,
            sequence_length,
            width,
            generator=generator,
            dtype=completion.full_projection.vectors.dtype,
        )
        changed = activations.clone()
        changed[:, -1] += 1
        with torch.no_grad():
            original_activation = completion(
                activations,
                prefix=f"{label}.completion",
            )
            changed_activation = completion(
                changed,
                prefix=f"{label}.completion",
            )
        torch.testing.assert_close(
            original_activation[:, :-1],
            changed_activation[:, :-1],
            rtol=0,
            atol=0,
        )


def _verify_optional_modal_completion(
    directory: Path,
    *,
    artifact_layer_index: int,
    checkpoint_hash: str,
    fisher_path: Path,
    bases: dict[str, FisherModeBasis],
    model: ToyTransformer,
    model_config: TransformerConfig,
    splits: AssociativeRecallSplits,
    manifest: dict[str, object],
    sequence_length: int,
) -> dict[str, object]:
    artifact_paths = modal_completion_artifact_paths(
        directory,
        artifact_layer_index,
    )
    input_path = artifact_paths.input_completion
    output_path = artifact_paths.output_completion
    report_path = artifact_paths.report_json
    markdown_path = artifact_paths.report_markdown
    bundle = (input_path, output_path, report_path, markdown_path)
    present = [path.is_file() for path in bundle]
    if not any(present):
        return {"present": False}
    if not all(present):
        missing = [
            path.name
            for path, exists in zip(bundle, present, strict=True)
            if not exists
        ]
        raise FileNotFoundError(
            f"incomplete modal completion artifacts: {missing}"
        )

    modal_executor_path = modal_executor_artifact_paths(
        directory,
        artifact_layer_index,
    ).executor
    if not modal_executor_path.is_file():
        raise FileNotFoundError(
            "modal completion requires its matching modal executor artifact"
        )
    report = _object(
        json.loads(report_path.read_text()),
        "modal completion report",
    )
    _assert_finite_tree(report, "modal completion report")
    expected_report_fields = {
        "format_version",
        "checkpoint_sha256",
        "fisher_sha256",
        "modal_executor_sha256",
        "input_completion_sha256",
        "output_completion_sha256",
        "teacher_state_sha256_before",
        "teacher_state_sha256_after",
        "teacher_was_frozen",
        "layer_index",
        "input_activation",
        "output_activation",
        "input_modes",
        "output_modes",
        "fit_protocol",
        "validation_candidates",
        "selected_configuration",
        "validation_ablations",
        "test_ablations",
        "local_completion_metrics",
        "validation_layer_output_fits",
        "accounting",
        "artifacts",
        "artifact_hashes_locked_before_test",
        "scientific_status",
        "elapsed_seconds",
    }
    if set(report) != expected_report_fields:
        raise ValueError("modal completion report fields mismatch")
    if _integer(
        report.get("format_version"),
        "modal completion format_version",
    ) != 1:
        raise ValueError("unsupported modal completion report format")

    fisher_hash = _sha256(fisher_path)
    modal_executor_hash = _sha256(modal_executor_path)
    input_hash = _sha256(input_path)
    output_hash = _sha256(output_path)
    expected_hashes = (
        ("checkpoint_sha256", checkpoint_hash),
        ("fisher_sha256", fisher_hash),
        ("modal_executor_sha256", modal_executor_hash),
        ("input_completion_sha256", input_hash),
        ("output_completion_sha256", output_hash),
    )
    for name, expected in expected_hashes:
        if report.get(name) != expected:
            raise ValueError(
                f"modal completion report {name} mismatch"
            )
    if report.get("artifact_hashes_locked_before_test") is not True:
        raise ValueError(
            "modal completion artifacts were not locked before test"
        )
    if report.get("teacher_was_frozen") is not True:
        raise ValueError("modal completion teacher was not frozen")
    if (
        report.get("scientific_status")
        != (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        )
    ):
        raise ValueError("modal completion scientific status mismatch")
    if _number(
        report.get("elapsed_seconds"),
        "modal completion elapsed_seconds",
    ) < 0:
        raise ValueError("modal completion elapsed_seconds is negative")
    expected_artifacts = {
        "input_completion": input_path.name,
        "output_completion": output_path.name,
        "modal_executor": modal_executor_path.name,
        "checkpoint": "checkpoint.pt",
        "fisher": fisher_path.name,
    }
    if _object(
        report.get("artifacts"),
        "modal completion artifacts",
    ) != expected_artifacts:
        raise ValueError("modal completion artifact names mismatch")
    if not markdown_path.read_text().strip():
        raise ValueError("modal completion Markdown report is empty")

    teacher_state_before = _module_state_sha256(model)
    if (
        report.get("teacher_state_sha256_before")
        != teacher_state_before
        or report.get("teacher_state_sha256_after")
        != teacher_state_before
    ):
        raise ValueError("modal completion teacher state hash mismatch")

    input_payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )
    output_payload = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_payload_fields = {
        "format_version",
        "artifact_kind",
        "config",
        "completion_state_dict",
        "metadata",
    }
    expected_state_fields = {
        "full_projection.position_mean",
        "full_projection.vectors",
        "graph.weight",
        "graph.bias",
    }
    for label, payload in (
        ("input", input_payload),
        ("output", output_payload),
    ):
        if not isinstance(payload, dict):
            raise ValueError(
                f"{label} modal completion artifact must be an object"
            )
        if set(payload) != expected_payload_fields:
            raise ValueError(
                f"{label} modal completion artifact fields mismatch"
            )
        if payload.get("format_version") != 1:
            raise ValueError(
                f"{label} modal completion artifact format mismatch"
            )
        if (
            payload.get("artifact_kind")
            != "position_conditioned_modal_completion"
        ):
            raise ValueError(
                f"{label} modal completion artifact kind mismatch"
            )
        state = _object(
            payload.get("completion_state_dict"),
            f"{label} modal completion state",
        )
        if set(state) != expected_state_fields:
            raise ValueError(
                f"{label} modal completion state fields mismatch"
            )

    input_completion, input_config, input_metadata = (
        load_position_modal_completion(input_path)
    )
    output_completion, output_config, output_metadata = (
        load_position_modal_completion(output_path)
    )
    protocol = _object(
        report.get("fit_protocol"),
        "modal completion fit_protocol",
    )
    expected_protocol_fields = {
        "fit_config",
        "fit_split",
        "selection_split",
        "coordinate_system",
        "target",
        "selection_rule",
        "validation_gate",
        "validation_is_fisher_informed",
        "test_used_for_fit_or_selection",
    }
    if set(protocol) != expected_protocol_fields:
        raise ValueError("modal completion fit protocol fields mismatch")
    if (
        protocol.get("fit_split") != "train"
        or protocol.get("selection_split") != "validation_fisher"
        or protocol.get("coordinate_system")
        != "full_position_centered_fisher_basis"
        or protocol.get("target") != "discarded_tail_coordinates"
        or protocol.get("selection_rule")
        != (
            "fewest_parameters_passing_behavior_gate_then_lowest_nll"
        )
        or protocol.get("validation_is_fisher_informed") is not True
        or protocol.get("test_used_for_fit_or_selection") is not False
    ):
        raise ValueError("modal completion fit protocol mismatch")
    fit_config_value = _object(
        protocol.get("fit_config"),
        "modal completion fit_config",
    )
    if set(fit_config_value) != {"ridge", "minimum_scale"}:
        raise ValueError("modal completion fit config fields mismatch")
    fit_config = ModalCompletionFitConfig(
        ridge=_number(
            fit_config_value.get("ridge"),
            "modal completion fit_config.ridge",
        ),
        minimum_scale=_number(
            fit_config_value.get("minimum_scale"),
            "modal completion fit_config.minimum_scale",
        ),
    )
    gate = _object(
        protocol.get("validation_gate"),
        "modal completion validation gate",
    )
    expected_gate_fields = {
        "minimum_answer_accuracy",
        "minimum_paired_accuracy",
        "maximum_nll_increase",
    }
    if set(gate) != expected_gate_fields:
        raise ValueError("modal completion validation gate fields mismatch")
    minimum_answer = _number(
        gate.get("minimum_answer_accuracy"),
        "modal completion gate.minimum_answer_accuracy",
    )
    minimum_paired = _number(
        gate.get("minimum_paired_accuracy"),
        "modal completion gate.minimum_paired_accuracy",
    )
    maximum_nll_increase = _number(
        gate.get("maximum_nll_increase"),
        "modal completion gate.maximum_nll_increase",
    )
    if (
        not 0 <= minimum_answer <= 1
        or not 0 <= minimum_paired <= 1
        or maximum_nll_increase < 0
    ):
        raise ValueError("modal completion validation gate is invalid")

    split_hashes: dict[str, str] = {}
    for name, split in (
        ("train", splits.train),
        ("validation_fisher", splits.validation),
        ("test", splits.test),
    ):
        section = _object(
            manifest.get(name),
            f"modal completion split manifest {name}",
        )
        ids = [
            _integer(
                value,
                f"modal completion split manifest {name}.context_ids",
            )
            for value in _array(
                section.get("context_ids"),
                (
                    "modal completion split manifest "
                    f"{name}.context_ids"
                ),
            )
        ]
        if ids != split.context_ids.tolist():
            raise ValueError(
                f"modal completion split context IDs mismatch: {name}"
            )
        split_hash = _tensor_sha256(split.context_ids)
        if section.get("context_ids_sha256") != split_hash:
            raise ValueError(
                f"modal completion split hash mismatch: {name}"
            )
        split_hashes[name] = split_hash

    layer_index = _integer(
        report.get("layer_index"),
        "modal completion layer_index",
    )
    if not 0 <= layer_index < len(model.layers):
        raise ValueError("modal completion layer index is outside the model")
    if layer_index != artifact_layer_index:
        raise ValueError(
            "modal completion layer index does not match artifact name"
        )
    input_name = (
        "layer.0.input"
        if layer_index == 0
        else f"layer.{layer_index - 1}.output"
    )
    output_name = f"layer.{layer_index}.output"
    if input_name not in bases or output_name not in bases:
        raise ValueError("modal completion Fisher bases are missing")

    selected_candidate_report = None
    expected_input_metadata_fields = {
        "checkpoint_sha256",
        "fisher_sha256",
        "modal_executor_sha256",
        "layer_index",
        "fit_context_ids_sha256",
        "selection_context_ids_sha256",
        "selected_candidate",
        "fit_protocol",
        "teacher_state_sha256",
        "boundary_role",
        "fit_activation",
    }
    expected_output_metadata_fields = {
        *expected_input_metadata_fields,
        "training_distribution",
    }
    for (
        label,
        metadata,
        expected_metadata_fields,
        fit_activation,
    ) in (
        (
            "input",
            input_metadata,
            expected_input_metadata_fields,
            input_name,
        ),
        (
            "output",
            output_metadata,
            expected_output_metadata_fields,
            output_name,
        ),
    ):
        if set(metadata) != expected_metadata_fields:
            raise ValueError(
                f"{label} modal completion metadata fields mismatch"
            )
        for name, expected in (
            ("checkpoint_sha256", checkpoint_hash),
            ("fisher_sha256", fisher_hash),
            ("modal_executor_sha256", modal_executor_hash),
            ("teacher_state_sha256", teacher_state_before),
            ("fit_context_ids_sha256", split_hashes["train"]),
            (
                "selection_context_ids_sha256",
                split_hashes["validation_fisher"],
            ),
        ):
            if metadata.get(name) != expected:
                raise ValueError(
                    f"{label} modal completion metadata {name} mismatch"
                )
        if (
            metadata.get("boundary_role") != label
            or metadata.get("fit_activation") != fit_activation
        ):
            raise ValueError(
                f"{label} modal completion boundary metadata mismatch"
            )
        if _json_normalized(metadata.get("fit_protocol")) != protocol:
            raise ValueError(
                f"{label} modal completion protocol metadata mismatch"
            )
        candidate_value = _json_normalized(
            metadata.get("selected_candidate")
        )
        if selected_candidate_report is None:
            selected_candidate_report = candidate_value
        elif candidate_value != selected_candidate_report:
            raise ValueError(
                "modal completion artifacts selected different candidates"
            )
    if (
        output_metadata.get("training_distribution")
        != "clean_frozen_teacher_output"
    ):
        raise ValueError(
            "output modal completion training distribution mismatch"
        )

    if (
        _integer(
            input_metadata.get("layer_index"),
            "input modal completion layer_index",
        )
        != layer_index
        or _integer(
            output_metadata.get("layer_index"),
            "output modal completion layer_index",
        )
        != layer_index
    ):
        raise ValueError("modal completion layer metadata mismatch")
    if (
        report.get("input_activation") != input_name
        or report.get("output_activation") != output_name
    ):
        raise ValueError("modal completion boundary names mismatch")
    input_modes = _integer(
        report.get("input_modes"),
        "modal completion input_modes",
    )
    output_modes = _integer(
        report.get("output_modes"),
        "modal completion output_modes",
    )
    input_basis = bases[input_name]
    output_basis = bases[output_name]
    if (
        not 1 <= input_modes < input_basis.width
        or not 1 <= output_modes < output_basis.width
    ):
        raise ValueError("modal completion mode count is invalid")

    selected_configuration = _object(
        report.get("selected_configuration"),
        "modal completion selected_configuration",
    )
    expected_selected_fields = {
        "input_graph_kind",
        "output_graph_kind",
        "ridge",
        "completion_learned_parameters",
        "validation_gate_passed",
    }
    if set(selected_configuration) != expected_selected_fields:
        raise ValueError(
            "modal completion selected configuration fields mismatch"
        )
    if not math.isclose(
        _number(
            selected_configuration.get("ridge"),
            "modal completion selected ridge",
        ),
        fit_config.ridge,
        rel_tol=0,
        abs_tol=0,
    ):
        raise ValueError("modal completion selected ridge mismatch")
    _validate_completion_bridge(
        input_completion,
        label="input",
        basis=input_basis,
        activation_name=input_name,
        sequence_length=sequence_length,
        width=model_config.d_model,
        kept_modes=input_modes,
        graph_kind=str(
            selected_configuration["input_graph_kind"]
        ),
    )
    _validate_completion_bridge(
        output_completion,
        label="output",
        basis=output_basis,
        activation_name=output_name,
        sequence_length=sequence_length,
        width=model_config.d_model,
        kept_modes=output_modes,
        graph_kind=str(
            selected_configuration["output_graph_kind"]
        ),
    )
    if (
        input_config != input_completion.config
        or output_config != output_completion.config
    ):
        raise ValueError("modal completion loaded config mismatch")

    for label, completion, basis in (
        ("input", input_completion, input_basis),
        ("output", output_completion, output_basis),
    ):
        generator = torch.Generator(device="cpu").manual_seed(
            79 if label == "input" else 83
        )
        coordinates = torch.randn(
            3,
            sequence_length,
            basis.width,
            generator=generator,
            dtype=completion.full_projection.vectors.dtype,
        )
        reconstructed = completion.full_projection.decode(coordinates)
        round_trip = completion.full_projection.encode(reconstructed)
        torch.testing.assert_close(
            round_trip,
            coordinates,
            rtol=2e-5,
            atol=2e-5,
        )

    train_activations = _collect_completion_activations(
        model,
        splits.train,
        (input_name, output_name),
    )
    validation_activations = _collect_completion_activations(
        model,
        splits.validation,
        (input_name, output_name),
    )
    test_activations = _collect_completion_activations(
        model,
        splits.test,
        (input_name, output_name),
    )
    completion_kinds = (
        ("shared_local_linear", True),
        ("position_local_linear", False),
    )
    input_candidates = {}
    output_candidates = {}
    for kind, shared_weights in completion_kinds:
        input_candidates[kind] = fit_local_modal_completion(
            train_activations[input_name],
            input_basis,
            kept_modes=input_modes,
            shared_weights=shared_weights,
            fit_config=fit_config,
        )
        output_candidates[kind] = fit_local_modal_completion(
            train_activations[output_name],
            output_basis,
            kept_modes=output_modes,
            shared_weights=shared_weights,
            fit_config=fit_config,
        )

    original_layer = model.layers[layer_index]
    teacher_validation_logits = associative_recall_answer_logits(
        model,
        splits.validation,
    )
    teacher_validation_metrics = (
        associative_recall_metrics_from_logits(
            splits.validation,
            teacher_validation_logits,
        )
    )
    reported_candidates = _array(
        report.get("validation_candidates"),
        "modal completion validation_candidates",
    )
    if len(reported_candidates) != 4:
        raise ValueError(
            "modal completion must report four validation candidates"
        )
    candidate_by_pair: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}
    recomputed_candidates: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}
    for input_kind, _ in completion_kinds:
        for output_kind, _ in completion_kinds:
            candidate_input, input_fit = input_candidates[input_kind]
            candidate_output, output_fit = output_candidates[
                output_kind
            ]
            executor = (
                PositionConditionedModalCompletionBottleneckExecutor(
                    original_layer,
                    input_completion=candidate_input,
                    output_completion=candidate_output,
                )
            )
            logits = _completion_replacement_logits(
                model,
                layer_index=layer_index,
                executor=executor,
                split=splits.validation,
            )
            behavior = _completion_behavior(
                splits.validation,
                logits,
                teacher_validation_logits,
            )
            metrics = _object(
                behavior["metrics"],
                "recomputed modal completion candidate metrics",
            )
            passed = (
                float(metrics["answer_accuracy"]) >= minimum_answer
                and float(metrics["paired_context_accuracy"])
                >= minimum_paired
                and float(metrics["hard_nll"])
                <= (
                    teacher_validation_metrics.hard_nll
                    + maximum_nll_increase
                )
            )
            recomputed_candidates[(input_kind, output_kind)] = {
                "input_graph_kind": input_kind,
                "output_graph_kind": output_kind,
                "input_fit": asdict(input_fit),
                "output_fit": asdict(output_fit),
                "completion_learned_parameters": (
                    candidate_input.graph.learned_parameter_count
                    + candidate_output.graph.learned_parameter_count
                ),
                "validation_behavior": behavior,
                "validation_gate_passed": passed,
            }
    for index, value in enumerate(reported_candidates):
        path = f"modal completion validation_candidates[{index}]"
        candidate = _object(value, path)
        expected_fields = {
            "input_graph_kind",
            "output_graph_kind",
            "input_fit",
            "output_fit",
            "completion_learned_parameters",
            "validation_behavior",
            "validation_gate_passed",
        }
        if set(candidate) != expected_fields:
            raise ValueError(f"{path} fields mismatch")
        pair = (
            str(candidate["input_graph_kind"]),
            str(candidate["output_graph_kind"]),
        )
        if pair in candidate_by_pair or pair not in recomputed_candidates:
            raise ValueError(
                "modal completion candidate pairs are invalid"
            )
        candidate_by_pair[pair] = candidate
        _assert_numeric_tree_match(
            candidate,
            recomputed_candidates[pair],
            path=path,
        )
    if set(candidate_by_pair) != set(recomputed_candidates):
        raise ValueError("modal completion candidate grid is incomplete")

    passing_pairs = [
        pair
        for pair, candidate in recomputed_candidates.items()
        if candidate["validation_gate_passed"]
    ]
    selection_pairs = (
        passing_pairs
        if passing_pairs
        else list(recomputed_candidates)
    )
    selected_pair = min(
        selection_pairs,
        key=lambda pair: (
            int(
                recomputed_candidates[pair][
                    "completion_learned_parameters"
                ]
            ),
            float(
                recomputed_candidates[pair][
                    "validation_behavior"
                ]["metrics"]["hard_nll"]  # type: ignore[index]
            ),
        ),
    )
    selected_expected = recomputed_candidates[selected_pair]
    expected_selected_configuration = {
        "input_graph_kind": selected_pair[0],
        "output_graph_kind": selected_pair[1],
        "ridge": fit_config.ridge,
        "completion_learned_parameters": selected_expected[
            "completion_learned_parameters"
        ],
        "validation_gate_passed": selected_expected[
            "validation_gate_passed"
        ],
    }
    _assert_numeric_tree_match(
        selected_configuration,
        expected_selected_configuration,
        path="modal completion selected_configuration",
    )
    if selected_candidate_report != _json_normalized(
        candidate_by_pair[selected_pair]
    ):
        raise ValueError(
            "modal completion selected candidate metadata mismatch"
        )
    for label, actual, expected in (
        (
            "input",
            input_completion,
            input_candidates[selected_pair[0]][0],
        ),
        (
            "output",
            output_completion,
            output_candidates[selected_pair[1]][0],
        ),
    ):
        actual_state = actual.state_dict()
        expected_state = expected.state_dict()
        if set(actual_state) != set(expected_state):
            raise ValueError(
                f"{label} modal completion selected state fields mismatch"
            )
        for name in actual_state:
            torch.testing.assert_close(
                actual_state[name],
                expected_state[name],
                rtol=0,
                atol=0,
            )

    mean_input = make_mean_modal_completion(
        train_activations[input_name],
        input_basis,
        kept_modes=input_modes,
    )
    mean_output = make_mean_modal_completion(
        train_activations[output_name],
        output_basis,
        kept_modes=output_modes,
    )
    full_input_projection = (
        PositionConditionedModalProjection.from_basis(
            input_basis,
            modes=input_basis.width,
        )
    )
    full_output_projection = (
        PositionConditionedModalProjection.from_basis(
            output_basis,
            modes=output_basis.width,
        )
    )
    kept_input_projection = (
        PositionConditionedModalProjection.from_basis(
            input_basis,
            modes=input_modes,
        )
    )
    kept_output_projection = (
        PositionConditionedModalProjection.from_basis(
            output_basis,
            modes=output_modes,
        )
    )
    saved_modal_executor, modal_config, modal_metadata = (
        load_position_modal_executor(modal_executor_path)
    )
    if (
        modal_config.input_modes != input_modes
        or modal_config.output_modes != output_modes
        or modal_metadata.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError(
            "modal completion base executor is incompatible"
        )
    systems = {
        "teacher": original_layer,
        "input_truncation": (
            PositionConditionedModalBottleneckExecutor(
                original_layer,
                kept_input_projection,
                full_output_projection,
            )
        ),
        "input_completion": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                input_completion=input_completion,
            )
        ),
        "output_truncation": (
            PositionConditionedModalBottleneckExecutor(
                original_layer,
                full_input_projection,
                kept_output_projection,
            )
        ),
        "output_completion": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                output_completion=output_completion,
            )
        ),
        "both_truncations": (
            PositionConditionedModalBottleneckExecutor(
                original_layer,
                kept_input_projection,
                kept_output_projection,
            )
        ),
        "mean_completion": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                input_completion=mean_input,
                output_completion=mean_output,
            )
        ),
        "both_completions": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                input_completion=input_completion,
                output_completion=output_completion,
            )
        ),
        "oracle_round_trip": (
            PositionConditionedModalBottleneckExecutor(
                original_layer,
                full_input_projection,
                full_output_projection,
            )
        ),
        "graph_zero_tail": saved_modal_executor,
        "graph_output_completion": (
            PositionConditionedCompletedModalGraphExecutor(
                saved_modal_executor,
                output_completion,
            )
        ),
    }
    expected_system_names = set(systems)
    teacher_test_logits = associative_recall_answer_logits(
        model,
        splits.test,
    )
    for split_name, split, teacher_logits in (
        (
            "validation",
            splits.validation,
            teacher_validation_logits,
        ),
        ("test", splits.test, teacher_test_logits),
    ):
        reported_ablations = _object(
            report.get(f"{split_name}_ablations"),
            f"modal completion {split_name}_ablations",
        )
        if set(reported_ablations) != expected_system_names:
            raise ValueError(
                f"modal completion {split_name} ablation names mismatch"
            )
        for name, executor in systems.items():
            logits = (
                teacher_logits
                if name == "teacher"
                else _completion_replacement_logits(
                    model,
                    layer_index=layer_index,
                    executor=executor,
                    split=split,
                )
            )
            expected_behavior = _completion_behavior(
                split,
                logits,
                teacher_logits,
            )
            _assert_numeric_tree_match(
                reported_ablations[name],
                expected_behavior,
                path=(
                    f"modal completion {split_name}_ablations."
                    f"{name}"
                ),
            )

    input_scale = _completion_train_tail_scale(
        train_activations[input_name],
        input_basis,
        kept_modes=input_modes,
        minimum_scale=fit_config.minimum_scale,
    )
    output_scale = _completion_train_tail_scale(
        train_activations[output_name],
        output_basis,
        kept_modes=output_modes,
        minimum_scale=fit_config.minimum_scale,
    )
    expected_local_metrics = {
        "input": {
            "validation": _completion_local_metrics(
                input_completion,
                mean_input,
                validation_activations[input_name],
                input_basis,
                train_tail_scale=input_scale,
            ),
            "test": _completion_local_metrics(
                input_completion,
                mean_input,
                test_activations[input_name],
                input_basis,
                train_tail_scale=input_scale,
            ),
        },
        "output": {
            "validation": _completion_local_metrics(
                output_completion,
                mean_output,
                validation_activations[output_name],
                output_basis,
                train_tail_scale=output_scale,
            ),
            "test": _completion_local_metrics(
                output_completion,
                mean_output,
                test_activations[output_name],
                output_basis,
                train_tail_scale=output_scale,
            ),
        },
    }
    _assert_numeric_tree_match(
        report.get("local_completion_metrics"),
        expected_local_metrics,
        path="modal completion local_completion_metrics",
    )

    expected_activation_fits = {
        name: _completion_activation_fit(
            executor,
            validation_activations[input_name],
            validation_activations[output_name],
            prefix=f"layer.{layer_index}",
        )
        for name, executor in systems.items()
        if name != "teacher"
    }
    _assert_numeric_tree_match(
        report.get("validation_layer_output_fits"),
        expected_activation_fits,
        path="modal completion validation_layer_output_fits",
    )

    input_parameters = (
        input_completion.graph.learned_parameter_count
    )
    output_parameters = (
        output_completion.graph.learned_parameter_count
    )
    completion_parameters = input_parameters + output_parameters
    map_multiplies = (
        input_completion.graph.edge_count
        + output_completion.graph.edge_count
    )
    tail_decode_multiplies = (
        sequence_length
        * model_config.d_model
        * (
            input_completion.tail_modes
            + output_completion.tail_modes
        )
    )
    completion_increment = (
        map_multiplies + tail_decode_multiplies
    )
    original_multiplies = (
        4
        * sequence_length
        * model_config.d_model
        * model_config.d_model
        + 2
        * sequence_length
        * sequence_length
        * model_config.d_model
        + 2
        * sequence_length
        * model_config.d_model
        * model_config.d_ff
    )
    graph_edges = saved_modal_executor.graph.edge_count
    base_graph_multiplies = (
        sequence_length
        * model_config.d_model
        * (input_modes + output_modes)
        + graph_edges
    )
    graph_output_increment = (
        output_completion.graph.edge_count
        + sequence_length
        * model_config.d_model
        * output_completion.tail_modes
    )
    completed_modal_graph_multiplies = (
        base_graph_multiplies + graph_output_increment
    )
    expected_accounting = {
        "completion_learned_parameters": completion_parameters,
        "input_completion_parameters": input_parameters,
        "output_completion_parameters": output_parameters,
        "completion_state_bytes": (
            _completion_state_bytes(input_completion)
            + _completion_state_bytes(output_completion)
        ),
        "completion_map_multiplies": map_multiplies,
        "additional_full_decode_multiplies": (
            tail_decode_multiplies
        ),
        "completion_incremental_multiplies": completion_increment,
        "original_block_estimated_multiplies": original_multiplies,
        "completed_bottleneck_estimated_multiplies": (
            original_multiplies
            + sequence_length
            * model_config.d_model
            * (
                input_modes
                + input_basis.width
                + output_modes
                + output_basis.width
            )
            + map_multiplies
        ),
        "base_modal_graph_estimated_multiplies": (
            base_graph_multiplies
        ),
        "completed_modal_graph_estimated_multiplies": (
            completed_modal_graph_multiplies
        ),
        "completed_graph_multiply_ratio": (
            completed_modal_graph_multiplies / original_multiplies
        ),
    }
    _assert_numeric_tree_match(
        report.get("accounting"),
        expected_accounting,
        path="modal completion accounting",
    )

    teacher_state_after = _module_state_sha256(model)
    if teacher_state_after != teacher_state_before:
        raise ValueError(
            "modal completion verification mutated the teacher"
        )
    if (
        _sha256(input_path) != input_hash
        or _sha256(output_path) != output_hash
    ):
        raise ValueError(
            "modal completion artifacts changed after their lock"
        )
    both_test = _object(
        _object(
            report.get("test_ablations"),
            "modal completion test_ablations",
        ).get("both_completions"),
        "modal completion test both_completions",
    )
    both_test_metrics = _object(
        both_test.get("metrics"),
        "modal completion test both_completions.metrics",
    )
    return {
        "present": True,
        "format_version": 1,
        "layer_index": layer_index,
        "input_modes": input_modes,
        "output_modes": output_modes,
        "input_graph_kind": input_completion.graph.graph_kind,
        "output_graph_kind": output_completion.graph.graph_kind,
        "candidate_count": len(reported_candidates),
        "behavior_system_count": len(systems),
        "learned_parameters": completion_parameters,
        "map_multiplies": map_multiplies,
        "test_accuracy": _number(
            both_test_metrics.get("answer_accuracy"),
            "modal completion test answer_accuracy",
        ),
        "test_paired_accuracy": _number(
            both_test_metrics.get("paired_context_accuracy"),
            "modal completion test paired_context_accuracy",
        ),
        "test_nll": _number(
            both_test_metrics.get("hard_nll"),
            "modal completion test hard_nll",
        ),
        "status": "verified",
    }


@torch.no_grad()
def _composition_replacement_logits(
    model: ToyTransformer,
    split,
    replacements: Mapping[int, LayerExecutor],
) -> torch.Tensor:
    originals = {
        layer_index: model.layers[layer_index]
        for layer_index in replacements
    }
    try:
        for layer_index, executor in replacements.items():
            model.replace_layer(layer_index, executor)
        return associative_recall_answer_logits(model, split)
    finally:
        for layer_index, original in originals.items():
            model.replace_layer(layer_index, original)


@torch.no_grad()
def _composition_replacement_activations(
    model: ToyTransformer,
    split,
    replacements: Mapping[int, LayerExecutor],
) -> dict[str, torch.Tensor]:
    originals = {
        layer_index: model.layers[layer_index]
        for layer_index in replacements
    }
    try:
        for layer_index, executor in replacements.items():
            model.replace_layer(layer_index, executor)
        return _collect_completion_activations(
            model,
            split,
            (
                "layer.0.output",
                "layer.1.input",
                "layer.1.output",
            ),
        )
    finally:
        for layer_index, original in originals.items():
            model.replace_layer(layer_index, original)


def _composition_behavior(
    split,
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> dict[str, object]:
    metrics = associative_recall_metrics_from_logits(split, logits)
    reference_probabilities = reference_logits.softmax(dim=-1)
    reference_log_probabilities = reference_logits.log_softmax(dim=-1)
    system_log_probabilities = logits.log_softmax(dim=-1)
    per_answer_kl = (
        reference_probabilities
        * (reference_log_probabilities - system_log_probabilities)
    ).sum(dim=-1)
    return {
        "metrics": asdict(metrics),
        "reference_to_system_answer_kl": per_answer_kl.mean().item(),
        "maximum_answer_kl": per_answer_kl.max().item(),
    }


def _composition_error_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor | None = None,
) -> float:
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    if matrix is None:
        numerator = (left64 * right64).sum()
        left_norm = left64.square().sum()
        right_norm = right64.square().sum()
    else:
        fisher = matrix.to(torch.float64)
        numerator = torch.einsum(
            "...i,ij,...j->",
            left64,
            fisher,
            right64,
        )
        left_norm = torch.einsum(
            "...i,ij,...j->",
            left64,
            fisher,
            left64,
        )
        right_norm = torch.einsum(
            "...i,ij,...j->",
            right64,
            fisher,
            right64,
        )
    denominator = (
        left_norm.clamp_min(0) * right_norm.clamp_min(0)
    ).sqrt()
    if denominator <= torch.finfo(torch.float64).eps:
        return 0.0
    return (numerator / denominator).item()


def _composition_error_metrics(
    error: torch.Tensor,
    fisher_matrix: torch.Tensor,
) -> dict[str, object]:
    values = error.to(torch.float64)
    fisher = fisher_matrix.to(torch.float64)
    fisher_energy = torch.einsum(
        "...i,ij,...j->...",
        values,
        fisher,
        values,
    ).clamp_min(0)
    return {
        "raw_rmse": values.square().mean().sqrt().item(),
        "raw_l2_per_token": (
            values.square().sum(dim=-1).mean().sqrt().item()
        ),
        "fisher_rms": fisher_energy.mean().sqrt().item(),
        "per_position_raw_rmse": (
            values.square().mean(dim=(0, 2)).sqrt().tolist()
        ),
        "per_position_fisher_rms": (
            fisher_energy.mean(dim=0).sqrt().tolist()
        ),
    }


def _composition_error_decomposition(
    teacher_output: torch.Tensor,
    upstream_output: torch.Tensor,
    composed_output: torch.Tensor,
    fisher_matrix: torch.Tensor,
) -> dict[str, object]:
    upstream_error = upstream_output - teacher_output
    local_error = composed_output - upstream_output
    total_error = composed_output - teacher_output
    identity_residual = total_error - (upstream_error + local_error)
    return {
        "definition": {
            "upstream": "B1(E0(h)) - B1(B0(h))",
            "local": "E1(E0(h)) - B1(E0(h))",
            "total": "E1(E0(h)) - B1(B0(h))",
        },
        "upstream": _composition_error_metrics(
            upstream_error,
            fisher_matrix,
        ),
        "local_same_input": _composition_error_metrics(
            local_error,
            fisher_matrix,
        ),
        "total": _composition_error_metrics(
            total_error,
            fisher_matrix,
        ),
        "upstream_local_raw_cosine": _composition_error_cosine(
            upstream_error,
            local_error,
        ),
        "upstream_local_fisher_cosine": _composition_error_cosine(
            upstream_error,
            local_error,
            fisher_matrix,
        ),
        "maximum_additive_identity_residual": (
            identity_residual.abs().max().item()
        ),
    }


def _composition_boundary_identity(
    activations: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    layer_0_output = activations["layer.0.output"]
    layer_1_input = activations["layer.1.input"]
    difference = layer_0_output - layer_1_input
    return {
        "exactly_equal": bool(
            torch.equal(layer_0_output, layer_1_input)
        ),
        "maximum_absolute_difference": difference.abs().max().item(),
    }


def _composition_evaluate_split(
    *,
    model: ToyTransformer,
    split,
    systems: Mapping[str, Mapping[int, LayerExecutor]],
    output_basis: FisherModeBasis,
) -> dict[str, object]:
    logits = {
        name: _composition_replacement_logits(
            model,
            split,
            replacements,
        )
        for name, replacements in systems.items()
    }
    teacher_logits = logits["teacher"]
    behavior = {
        name: _composition_behavior(
            split,
            system_logits,
            teacher_logits,
        )
        for name, system_logits in logits.items()
    }
    activation_system_names = (
        "teacher",
        "layer_0_completed",
        "layer_1_completed",
        "both_completed",
    )
    activations = {
        name: _composition_replacement_activations(
            model,
            split,
            systems[name],
        )
        for name in activation_system_names
    }
    clean_local_error = (
        activations["layer_1_completed"]["layer.1.output"]
        - activations["teacher"]["layer.1.output"]
    )
    shifted_local_error = (
        activations["both_completed"]["layer.1.output"]
        - activations["layer_0_completed"]["layer.1.output"]
    )
    return {
        "systems_vs_teacher": behavior,
        "same_input_contracts": {
            "clean_input": {
                "reference_system": "teacher",
                "compiled_system": "layer_1_completed",
                "suffix_behavior": _composition_behavior(
                    split,
                    logits["layer_1_completed"],
                    logits["teacher"],
                ),
                "layer_output_error": _composition_error_metrics(
                    clean_local_error,
                    output_basis.matrix,
                ),
            },
            "compiled_layer_0_input": {
                "reference_system": "layer_0_completed",
                "compiled_system": "both_completed",
                "suffix_behavior": _composition_behavior(
                    split,
                    logits["both_completed"],
                    logits["layer_0_completed"],
                ),
                "layer_output_error": _composition_error_metrics(
                    shifted_local_error,
                    output_basis.matrix,
                ),
            },
        },
        "error_decomposition": _composition_error_decomposition(
            activations["teacher"]["layer.1.output"],
            activations["layer_0_completed"]["layer.1.output"],
            activations["both_completed"]["layer.1.output"],
            output_basis.matrix,
        ),
        "boundary_identity": {
            name: _composition_boundary_identity(values)
            for name, values in activations.items()
        },
    }


def _composition_validation_gate_passed(
    evaluation: Mapping[str, object],
    gate: Mapping[str, float],
) -> bool:
    systems = _object(
        evaluation.get("systems_vs_teacher"),
        "composition systems_vs_teacher",
    )
    contracts = _object(
        evaluation.get("same_input_contracts"),
        "composition same_input_contracts",
    )
    for name in ("layer_1_completed", "both_completed"):
        behavior = _object(
            systems.get(name),
            f"composition systems_vs_teacher.{name}",
        )
        metrics = _object(
            behavior.get("metrics"),
            f"composition systems_vs_teacher.{name}.metrics",
        )
        if (
            float(metrics["answer_accuracy"])
            < gate["minimum_answer_accuracy"]
            or float(metrics["paired_context_accuracy"])
            < gate["minimum_paired_accuracy"]
        ):
            return False
    for name in ("clean_input", "compiled_layer_0_input"):
        contract = _object(
            contracts.get(name),
            f"composition same_input_contracts.{name}",
        )
        suffix = _object(
            contract.get("suffix_behavior"),
            f"composition same_input_contracts.{name}.suffix_behavior",
        )
        metrics = _object(
            suffix.get("metrics"),
            (
                "composition same_input_contracts."
                f"{name}.suffix_behavior.metrics"
            ),
        )
        reference_name = str(contract.get("reference_system"))
        reference = _object(
            systems.get(reference_name),
            f"composition systems_vs_teacher.{reference_name}",
        )
        reference_metrics = _object(
            reference.get("metrics"),
            f"composition systems_vs_teacher.{reference_name}.metrics",
        )
        if (
            float(metrics["hard_nll"])
            > (
                float(reference_metrics["hard_nll"])
                + gate["maximum_same_input_nll_increase"]
            )
            or float(suffix["reference_to_system_answer_kl"])
            > gate["maximum_same_input_answer_kl"]
        ):
            return False
    return True


def _composition_layer_accounting(
    *,
    executor: PositionConditionedModalGraphExecutor,
    completed: PositionConditionedCompletedModalGraphExecutor,
    sequence_length: int,
    width: int,
) -> dict[str, object]:
    input_modes = executor.input_projection.modes
    output_modes = executor.output_projection.modes
    routing_width = executor.graph.hidden_modes
    base_multiplies = (
        sequence_length
        * width
        * (input_modes + output_modes)
        + executor.graph.edge_count
    )
    output_completion = completed.output_completion
    parameter_tensors = list(completed.parameters())
    buffer_tensors = list(completed.buffers())
    parameter_elements = sum(
        tensor.numel() for tensor in parameter_tensors
    )
    buffer_elements = sum(
        tensor.numel() for tensor in buffer_tensors
    )
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in parameter_tensors
    )
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in buffer_tensors
    )
    completion_increment = (
        output_completion.graph.edge_count
        + sequence_length * width * output_completion.tail_modes
    )
    return {
        "input_modes": input_modes,
        "routing_width": routing_width,
        "output_modes": output_modes,
        "output_tail_modes": output_completion.tail_modes,
        "graph_edges": executor.graph.edge_count,
        "graph_learned_parameters": sum(
            parameter.numel()
            for parameter in executor.graph.parameters()
        ),
        "output_completion_learned_parameters": (
            output_completion.graph.learned_parameter_count
        ),
        "base_graph_estimated_multiplies": base_multiplies,
        "output_completion_incremental_multiplies": (
            completion_increment
        ),
        "completed_estimated_multiplies": (
            base_multiplies + completion_increment
        ),
        "storage": {
            "learned_parameter_elements": parameter_elements,
            "learned_parameter_bytes": parameter_bytes,
            "stored_buffer_elements": buffer_elements,
            "stored_buffer_bytes": buffer_bytes,
            "total_state_elements": (
                parameter_elements + buffer_elements
            ),
            "total_state_bytes": parameter_bytes + buffer_bytes,
        },
    }


def _load_composition_layer(
    *,
    layer_index: int,
    executor_path: Path,
    output_completion_path: Path,
    checkpoint_hash: str,
    fisher_hash: str,
    teacher_state_hash: str,
    bases: Mapping[str, FisherModeBasis],
    input_activation: str,
    output_activation: str,
    sequence_length: int,
    width: int,
    fit_context_ids_sha256: str,
    selection_context_ids_sha256: str,
) -> tuple[
    PositionConditionedModalGraphExecutor,
    PositionConditionedCompletedModalGraphExecutor,
    dict[str, object],
]:
    executor_hash = _sha256(executor_path)
    output_completion_hash = _sha256(output_completion_path)
    executor, config, executor_metadata = (
        load_position_modal_executor(executor_path)
    )
    output_completion, completion_config, completion_metadata = (
        load_position_modal_completion(output_completion_path)
    )
    if (
        config.input_activation != input_activation
        or config.output_activation != output_activation
        or config.sequence_length != sequence_length
    ):
        raise ValueError(
            f"modal composition layer {layer_index} executor boundary mismatch"
        )
    if (
        input_activation not in bases
        or output_activation not in bases
    ):
        raise ValueError(
            f"modal composition layer {layer_index} Fisher basis missing"
        )
    if (
        config.input_modes <= 0
        or config.input_modes > bases[input_activation].width
        or config.output_modes <= 0
        or config.output_modes > bases[output_activation].width
    ):
        raise ValueError(
            f"modal composition layer {layer_index} mode count mismatch"
        )
    if (
        completion_config.activation_name != output_activation
        or completion_config.sequence_length != sequence_length
        or completion_config.width != width
        or completion_config.kept_modes != config.output_modes
    ):
        raise ValueError(
            f"modal composition layer {layer_index} completion boundary mismatch"
        )
    expected_executor_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "layer_index": layer_index,
        "teacher_state_sha256": teacher_state_hash,
        "teacher_was_frozen": True,
        "training_distribution": (
            "clean_frozen_teacher_boundary_pairs"
        ),
        "training_contract": "same_forward_input_output",
        "target": "frozen_teacher_layer_output_for_exact_input",
        "robustification_used": False,
        "compensation_target_used": False,
        "fit_split": "train",
        "selection_split": "validation_fisher",
        "test_used_for_fit_or_selection": False,
        "fit_context_ids_sha256": fit_context_ids_sha256,
        "selection_context_ids_sha256": (
            selection_context_ids_sha256
        ),
    }
    for name, expected in expected_executor_metadata.items():
        if executor_metadata.get(name) != expected:
            raise ValueError(
                "modal composition layer "
                f"{layer_index} executor {name} mismatch"
            )
    expected_completion_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "modal_executor_sha256": executor_hash,
        "layer_index": layer_index,
        "teacher_state_sha256": teacher_state_hash,
        "fit_context_ids_sha256": fit_context_ids_sha256,
        "selection_context_ids_sha256": (
            selection_context_ids_sha256
        ),
        "boundary_role": "output",
        "fit_activation": output_activation,
        "training_distribution": "clean_frozen_teacher_output",
    }
    for name, expected in expected_completion_metadata.items():
        if completion_metadata.get(name) != expected:
            raise ValueError(
                "modal composition layer "
                f"{layer_index} output completion {name} mismatch"
            )
    completion_protocol = _object(
        completion_metadata.get("fit_protocol"),
        f"modal composition layer {layer_index} completion fit_protocol",
    )
    if (
        completion_protocol.get("fit_split") != "train"
        or completion_protocol.get("selection_split")
        != "validation_fisher"
        or completion_protocol.get("test_used_for_fit_or_selection")
        is not False
    ):
        raise ValueError(
            f"modal composition layer {layer_index} fit provenance mismatch"
        )

    for label, projection, basis in (
        (
            "input",
            executor.input_projection,
            bases[input_activation],
        ),
        (
            "output",
            executor.output_projection,
            bases[output_activation],
        ),
        (
            "completion",
            output_completion.full_projection,
            bases[output_activation],
        ),
    ):
        if basis.position_means is None:
            raise ValueError(
                "modal composition layer "
                f"{layer_index} {label} basis lacks position means"
            )
        torch.testing.assert_close(
            projection.position_mean,
            basis.position_means.to(projection.position_mean),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            projection.vectors,
            basis.vectors[
                :,
                : projection.modes,
            ].to(projection.vectors),
            rtol=0,
            atol=0,
        )

    completed = PositionConditionedCompletedModalGraphExecutor(
        executor,
        output_completion,
    )
    if any(
        isinstance(module, TransformerBlock)
        for module in completed.modules()
    ):
        raise ValueError(
            f"modal composition layer {layer_index} contains a transformer"
        )
    provenance = {
        "executor_sha256": executor_hash,
        "output_completion_sha256": output_completion_hash,
        "input_activation": input_activation,
        "output_activation": output_activation,
        "input_modes": config.input_modes,
        "output_modes": config.output_modes,
        "routing_width": config.routing_width,
        "teacher_state_sha256": teacher_state_hash,
        "training_distribution": executor_metadata[
            "training_distribution"
        ],
        "training_contract": executor_metadata["training_contract"],
        "target": executor_metadata["target"],
        "robustification_used": executor_metadata[
            "robustification_used"
        ],
        "compensation_target_used": executor_metadata[
            "compensation_target_used"
        ],
        "test_used_for_fit_or_selection": bool(
            executor_metadata["test_used_for_fit_or_selection"]
            or completion_protocol["test_used_for_fit_or_selection"]
        ),
        "contains_transformer_block": False,
    }
    return executor, completed, provenance


def _verify_optional_modal_composition(
    directory: Path,
    *,
    checkpoint_hash: str,
    fisher_path: Path,
    bases: Mapping[str, FisherModeBasis],
    model: ToyTransformer,
    model_config: TransformerConfig,
    splits: AssociativeRecallSplits,
    sequence_length: int,
) -> dict[str, object]:
    report_path = directory / "modal_composition_report.json"
    markdown_path = directory / "modal_composition_report.md"
    present = (report_path.is_file(), markdown_path.is_file())
    if not any(present):
        return {"present": False}
    if not all(present):
        missing = [
            path.name
            for path, exists in zip(
                (report_path, markdown_path),
                present,
                strict=True,
            )
            if not exists
        ]
        raise FileNotFoundError(
            f"incomplete modal composition reports: {missing}"
        )
    if len(model.layers) != 2:
        raise ValueError(
            "modal composition report requires exactly two model layers"
        )

    report = _object(
        json.loads(report_path.read_text()),
        "modal composition report",
    )
    _assert_finite_tree(report, "modal composition report")
    expected_report_fields = {
        "format_version",
        "checkpoint_sha256",
        "fisher_sha256",
        "teacher_state_sha256_before",
        "teacher_state_sha256_after",
        "teacher_was_frozen",
        "protocol",
        "layer_provenance",
        "validation",
        "test",
        "accounting",
        "artifacts",
        "artifact_hashes_locked_before_validation_and_test",
        "scientific_status",
        "elapsed_seconds",
    }
    if set(report) != expected_report_fields:
        raise ValueError("modal composition report fields mismatch")
    if _integer(
        report.get("format_version"),
        "modal composition format_version",
    ) != 1:
        raise ValueError("unsupported modal composition report format")
    fisher_hash = _sha256(fisher_path)
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("modal composition checkpoint hash mismatch")
    if report.get("fisher_sha256") != fisher_hash:
        raise ValueError("modal composition Fisher hash mismatch")
    if report.get("teacher_was_frozen") is not True:
        raise ValueError("modal composition teacher was not frozen")
    if (
        report.get("artifact_hashes_locked_before_validation_and_test")
        is not True
    ):
        raise ValueError(
            "modal composition artifacts were not locked before evaluation"
        )
    if (
        report.get("scientific_status")
        != (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        )
    ):
        raise ValueError("modal composition scientific status mismatch")
    if _number(
        report.get("elapsed_seconds"),
        "modal composition elapsed_seconds",
    ) < 0:
        raise ValueError("modal composition elapsed_seconds is negative")
    if not markdown_path.read_text().strip():
        raise ValueError("modal composition Markdown report is empty")

    teacher_state_before = _module_state_sha256(model)
    if (
        report.get("teacher_state_sha256_before")
        != teacher_state_before
        or report.get("teacher_state_sha256_after")
        != teacher_state_before
    ):
        raise ValueError("modal composition teacher state hash mismatch")

    executor_paths = {
        layer_index: modal_executor_artifact_paths(
            directory,
            layer_index,
        ).executor
        for layer_index in (0, 1)
    }
    completion_paths = {
        layer_index: modal_completion_artifact_paths(
            directory,
            layer_index,
        ).output_completion
        for layer_index in (0, 1)
    }
    runtime_paths = {
        "layer_0_executor": executor_paths[0],
        "layer_0_output_completion": completion_paths[0],
        "layer_1_executor": executor_paths[1],
        "layer_1_output_completion": completion_paths[1],
    }
    missing_runtime = [
        path.name
        for path in runtime_paths.values()
        if not path.is_file()
    ]
    if missing_runtime:
        raise FileNotFoundError(
            f"incomplete modal composition artifacts: {missing_runtime}"
        )
    hashes_before = {
        name: _sha256(path)
        for name, path in runtime_paths.items()
    }
    expected_artifacts = {
        name: {
            "filename": runtime_paths[name].name,
            "sha256": hashes_before[name],
        }
        for name in runtime_paths
    }
    _assert_numeric_tree_match(
        report.get("artifacts"),
        expected_artifacts,
        path="modal composition artifacts",
    )

    fit_context_ids_sha256 = _tensor_sha256(
        splits.train.context_ids
    )
    selection_context_ids_sha256 = _tensor_sha256(
        splits.validation.context_ids
    )
    layer_0, completed_0, provenance_0 = _load_composition_layer(
        layer_index=0,
        executor_path=executor_paths[0],
        output_completion_path=completion_paths[0],
        checkpoint_hash=checkpoint_hash,
        fisher_hash=fisher_hash,
        teacher_state_hash=teacher_state_before,
        bases=bases,
        input_activation="layer.0.input",
        output_activation="layer.0.output",
        sequence_length=sequence_length,
        width=model_config.d_model,
        fit_context_ids_sha256=fit_context_ids_sha256,
        selection_context_ids_sha256=(
            selection_context_ids_sha256
        ),
    )
    layer_1, completed_1, provenance_1 = _load_composition_layer(
        layer_index=1,
        executor_path=executor_paths[1],
        output_completion_path=completion_paths[1],
        checkpoint_hash=checkpoint_hash,
        fisher_hash=fisher_hash,
        teacher_state_hash=teacher_state_before,
        bases=bases,
        input_activation="layer.0.output",
        output_activation="layer.1.output",
        sequence_length=sequence_length,
        width=model_config.d_model,
        fit_context_ids_sha256=fit_context_ids_sha256,
        selection_context_ids_sha256=(
            selection_context_ids_sha256
        ),
    )
    expected_provenance = {
        "layer_0": provenance_0,
        "layer_1_pristine": provenance_1,
    }
    _assert_numeric_tree_match(
        report.get("layer_provenance"),
        expected_provenance,
        path="modal composition layer_provenance",
    )

    validation_gate = {
        "minimum_answer_accuracy": 0.995,
        "minimum_paired_accuracy": 0.99,
        "maximum_same_input_nll_increase": 0.01,
        "maximum_same_input_answer_kl": 0.01,
    }
    expected_protocol = {
        "layer_0_training_distribution": (
            provenance_0["training_distribution"]
        ),
        "layer_0_training_contract": provenance_0[
            "training_contract"
        ],
        "layer_0_target": provenance_0["target"],
        "layer_1_training_distribution": (
            provenance_1["training_distribution"]
        ),
        "layer_1_training_contract": provenance_1[
            "training_contract"
        ],
        "layer_1_target": provenance_1["target"],
        "layer_1_robustification_used": False,
        "forbidden_compensation_pair_used": False,
        "validation_split": "validation_fisher",
        "evaluation_split": "test",
        "test_used_for_fit_or_selection": bool(
            provenance_0["test_used_for_fit_or_selection"]
            or provenance_1["test_used_for_fit_or_selection"]
        ),
        "validation_gate": validation_gate,
        "validation_gate_passed": True,
    }
    _assert_numeric_tree_match(
        report.get("protocol"),
        expected_protocol,
        path="modal composition protocol",
    )

    systems: dict[str, dict[int, LayerExecutor]] = {
        "teacher": {},
        "layer_0_zero_tail": {0: layer_0},
        "layer_1_zero_tail": {1: layer_1},
        "both_zero_tail": {0: layer_0, 1: layer_1},
        "layer_0_completed": {0: completed_0},
        "layer_1_completed": {1: completed_1},
        "both_completed": {0: completed_0, 1: completed_1},
        "layer_0_completed_layer_1_zero_tail": {
            0: completed_0,
            1: layer_1,
        },
        "layer_0_zero_tail_layer_1_completed": {
            0: layer_0,
            1: completed_1,
        },
    }
    validation = _composition_evaluate_split(
        model=model,
        split=splits.validation,
        systems=systems,
        output_basis=bases["layer.1.output"],
    )
    if not _composition_validation_gate_passed(
        validation,
        validation_gate,
    ):
        raise ValueError(
            "modal composition recomputed validation gate failed"
        )
    _assert_numeric_tree_match(
        report.get("validation"),
        validation,
        path="modal composition validation",
    )
    test = _composition_evaluate_split(
        model=model,
        split=splits.test,
        systems=systems,
        output_basis=bases["layer.1.output"],
    )
    _assert_numeric_tree_match(
        report.get("test"),
        test,
        path="modal composition test",
    )

    for split_name, evaluation in (
        ("validation", validation),
        ("test", test),
    ):
        identities = _object(
            evaluation.get("boundary_identity"),
            f"modal composition {split_name}.boundary_identity",
        )
        if any(
            _object(
                value,
                (
                    f"modal composition {split_name}."
                    f"boundary_identity.{name}"
                ),
            ).get("exactly_equal")
            is not True
            for name, value in identities.items()
        ):
            raise ValueError(
                f"modal composition {split_name} boundary identity failed"
            )

    layer_0_accounting = _composition_layer_accounting(
        executor=layer_0,
        completed=completed_0,
        sequence_length=sequence_length,
        width=model_config.d_model,
    )
    layer_1_accounting = _composition_layer_accounting(
        executor=layer_1,
        completed=completed_1,
        sequence_length=sequence_length,
        width=model_config.d_model,
    )
    original_block_multiplies = (
        4
        * sequence_length
        * model_config.d_model
        * model_config.d_model
        + 2
        * sequence_length
        * sequence_length
        * model_config.d_model
        + 2
        * sequence_length
        * model_config.d_model
        * model_config.d_ff
    )
    compiled_multiplies = (
        int(layer_0_accounting["completed_estimated_multiplies"])
        + int(layer_1_accounting["completed_estimated_multiplies"])
    )
    layer_storage = (
        _object(
            layer_0_accounting["storage"],
            "modal composition layer_0 storage",
        ),
        _object(
            layer_1_accounting["storage"],
            "modal composition layer_1 storage",
        ),
    )
    compiled_parameter_elements = sum(
        int(storage["learned_parameter_elements"])
        for storage in layer_storage
    )
    compiled_parameter_bytes = sum(
        int(storage["learned_parameter_bytes"])
        for storage in layer_storage
    )
    compiled_buffer_elements = sum(
        int(storage["stored_buffer_elements"])
        for storage in layer_storage
    )
    compiled_buffer_bytes = sum(
        int(storage["stored_buffer_bytes"])
        for storage in layer_storage
    )
    expected_accounting = {
        "layer_0": layer_0_accounting,
        "layer_1": layer_1_accounting,
        "original_block_estimated_multiplies": (
            original_block_multiplies
        ),
        "original_two_block_estimated_multiplies": (
            2 * original_block_multiplies
        ),
        "compiled_two_layer_estimated_multiplies": compiled_multiplies,
        "compiled_multiply_ratio": (
            compiled_multiplies / (2 * original_block_multiplies)
        ),
        "compiled_learned_parameter_elements": (
            compiled_parameter_elements
        ),
        "compiled_learned_parameter_bytes": compiled_parameter_bytes,
        "compiled_stored_buffer_elements": compiled_buffer_elements,
        "compiled_stored_buffer_bytes": compiled_buffer_bytes,
    }
    _assert_numeric_tree_match(
        report.get("accounting"),
        expected_accounting,
        path="modal composition accounting",
    )

    if {
        name: _sha256(path)
        for name, path in runtime_paths.items()
    } != hashes_before:
        raise ValueError(
            "modal composition runtime artifacts changed during verification"
        )
    if _module_state_sha256(model) != teacher_state_before:
        raise ValueError(
            "modal composition verification mutated the teacher"
        )

    both_test = _object(
        _object(
            test.get("systems_vs_teacher"),
            "modal composition test.systems_vs_teacher",
        ).get("both_completed"),
        "modal composition test.systems_vs_teacher.both_completed",
    )
    both_test_metrics = _object(
        both_test.get("metrics"),
        (
            "modal composition test.systems_vs_teacher."
            "both_completed.metrics"
        ),
    )
    shifted_contract = _object(
        _object(
            validation.get("same_input_contracts"),
            "modal composition validation.same_input_contracts",
        ).get("compiled_layer_0_input"),
        (
            "modal composition validation.same_input_contracts."
            "compiled_layer_0_input"
        ),
    )
    shifted_suffix = _object(
        shifted_contract.get("suffix_behavior"),
        (
            "modal composition validation.same_input_contracts."
            "compiled_layer_0_input.suffix_behavior"
        ),
    )
    return {
        "present": True,
        "format_version": 1,
        "layer_count": 2,
        "behavior_system_count": len(systems),
        "validation_gate_passed": True,
        "compiled_estimated_multiplies": compiled_multiplies,
        "compiled_multiply_ratio": expected_accounting[
            "compiled_multiply_ratio"
        ],
        "validation_shifted_same_input_kl": _number(
            shifted_suffix.get("reference_to_system_answer_kl"),
            "modal composition validation shifted same-input KL",
        ),
        "test_accuracy": _number(
            both_test_metrics.get("answer_accuracy"),
            "modal composition test answer_accuracy",
        ),
        "test_paired_accuracy": _number(
            both_test_metrics.get("paired_context_accuracy"),
            "modal composition test paired_context_accuracy",
        ),
        "test_nll": _number(
            both_test_metrics.get("hard_nll"),
            "modal composition test hard_nll",
        ),
        "status": "verified",
    }


def _fused_freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _fused_state_storage(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    trainable = [
        parameter
        for parameter in parameters
        if parameter.requires_grad
    ]

    def elements(tensors: list[torch.Tensor]) -> int:
        return sum(tensor.numel() for tensor in tensors)

    def size_bytes(tensors: list[torch.Tensor]) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )

    return {
        "parameter_elements": elements(parameters),
        "parameter_bytes": size_bytes(parameters),
        "trainable_parameter_elements": elements(trainable),
        "trainable_parameter_bytes": size_bytes(trainable),
        "buffer_elements": elements(buffers),
        "buffer_bytes": size_bytes(buffers),
        "total_state_elements": elements(parameters) + elements(buffers),
        "total_state_bytes": size_bytes(parameters) + size_bytes(buffers),
    }


def _fused_answer_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    reference = reference_logits.to(torch.float64)
    candidate = candidate_logits.to(torch.float64)
    reference_probabilities = reference.softmax(dim=-1)
    return (
        reference_probabilities
        * (
            reference.log_softmax(dim=-1)
            - candidate.log_softmax(dim=-1)
        )
    ).sum(dim=-1)


def _fused_comparison(
    *,
    split,
    unfused_logits: torch.Tensor,
    fused_logits: torch.Tensor,
) -> dict[str, object]:
    unfused_metrics = associative_recall_metrics_from_logits(
        split,
        unfused_logits,
    )
    fused_metrics = associative_recall_metrics_from_logits(
        split,
        fused_logits,
    )
    unfused_predictions = unfused_logits.argmax(dim=-1)
    fused_predictions = fused_logits.argmax(dim=-1)
    answer_kl = _fused_answer_kl(unfused_logits, fused_logits)
    return {
        "answer_accuracy_exactly_equal": (
            fused_metrics.answer_accuracy
            == unfused_metrics.answer_accuracy
        ),
        "paired_context_accuracy_exactly_equal": (
            fused_metrics.paired_context_accuracy
            == unfused_metrics.paired_context_accuracy
        ),
        "argmax_predictions_exactly_equal": bool(
            torch.equal(fused_predictions, unfused_predictions)
        ),
        "unfused_argmax_sha256": _tensor_sha256(
            unfused_predictions
        ),
        "fused_argmax_sha256": _tensor_sha256(fused_predictions),
        "absolute_hard_nll_delta": abs(
            fused_metrics.hard_nll - unfused_metrics.hard_nll
        ),
        "mean_unfused_to_fused_answer_kl": answer_kl.mean().item(),
        "maximum_unfused_to_fused_answer_kl": answer_kl.max().item(),
        "maximum_answer_logit_difference": (
            (fused_logits - unfused_logits).abs().max().item()
        ),
    }


def _fused_evaluate_split(
    *,
    split,
    teacher: ToyTransformer,
    unfused: ToyTransformer,
    monolithic: FusedToyTransformer,
    lazy: FusedToyTransformer,
) -> dict[str, object]:
    logits = {
        "teacher": associative_recall_answer_logits(teacher, split),
        "unfused": associative_recall_answer_logits(unfused, split),
        "monolithic": associative_recall_answer_logits(
            monolithic,
            split,
        ),
        "lazy": associative_recall_answer_logits(lazy, split),
    }
    return {
        "systems": {
            name: asdict(
                associative_recall_metrics_from_logits(split, values)
            )
            for name, values in logits.items()
        },
        "monolithic_vs_unfused": _fused_comparison(
            split=split,
            unfused_logits=logits["unfused"],
            fused_logits=logits["monolithic"],
        ),
        "lazy_vs_unfused": _fused_comparison(
            split=split,
            unfused_logits=logits["unfused"],
            fused_logits=logits["lazy"],
        ),
        "lazy_vs_monolithic": {
            "logits_bit_exact": bool(
                torch.equal(logits["lazy"], logits["monolithic"])
            ),
            "maximum_logit_difference": (
                logits["lazy"] - logits["monolithic"]
            ).abs().max().item(),
            "monolithic_argmax_sha256": _tensor_sha256(
                logits["monolithic"].argmax(dim=-1)
            ),
            "lazy_argmax_sha256": _tensor_sha256(
                logits["lazy"].argmax(dim=-1)
            ),
        },
    }


def _fused_gate_passed(
    comparison_value: object,
    gate: Mapping[str, float],
) -> bool:
    comparison = _object(
        comparison_value,
        "fused comparison",
    )
    return bool(
        comparison.get("answer_accuracy_exactly_equal")
        and comparison.get("paired_context_accuracy_exactly_equal")
        and comparison.get("argmax_predictions_exactly_equal")
        and _number(
            comparison.get("absolute_hard_nll_delta"),
            "fused absolute NLL delta",
        )
        <= gate["maximum_absolute_nll_delta"]
        and _number(
            comparison.get("mean_unfused_to_fused_answer_kl"),
            "fused mean answer KL",
        )
        <= gate["maximum_mean_answer_kl"]
        and _number(
            comparison.get("maximum_answer_logit_difference"),
            "fused maximum answer-logit difference",
        )
        <= gate["maximum_answer_logit_difference"]
    )


def _fused_expected_arithmetic(
    *,
    stack: FusedTwoLayerModalStack,
    model_config: TransformerConfig,
    completed_layers: tuple[
        PositionConditionedCompletedModalGraphExecutor,
        PositionConditionedCompletedModalGraphExecutor,
    ],
) -> dict[str, object]:
    sequence_length = stack.config.first.sequence_length
    width = stack.config.first.width
    first_routing = stack.config.first.routing_width
    second_routing = stack.config.second.routing_width
    causal_pairs = sequence_length * (sequence_length + 1) // 2
    dense_components = {
        "input_to_layer_0_hidden": (
            sequence_length
            * sequence_length
            * width
            * first_routing
        ),
        "layer_0_hidden_to_layer_1_hidden": (
            sequence_length
            * sequence_length
            * first_routing
            * second_routing
        ),
        "layer_1_hidden_to_residual_output": (
            sequence_length * second_routing * width
        ),
    }
    triangular_components = {
        "input_to_layer_0_hidden": (
            causal_pairs * width * first_routing
        ),
        "layer_0_hidden_to_layer_1_hidden": (
            causal_pairs * first_routing * second_routing
        ),
        "layer_1_hidden_to_residual_output": (
            sequence_length * second_routing * width
        ),
    }
    logical_multiplies = 0
    for completed in completed_layers:
        executor = completed.base_executor
        input_modes = executor.input_projection.modes
        output_modes = executor.output_projection.modes
        routing_width = executor.graph.hidden_modes
        base_multiplies = (
            sequence_length
            * width
            * (input_modes + output_modes)
            + causal_pairs * input_modes * routing_width
            + sequence_length * routing_width * output_modes
        )
        tail_modes = completed.output_completion.tail_modes
        completion_multiplies = (
            sequence_length * output_modes * tail_modes
            + sequence_length * width * tail_modes
        )
        logical_multiplies += base_multiplies + completion_multiplies
    original_per_block = (
        4 * sequence_length * width * width
        + 2 * sequence_length * sequence_length * width
        + 2
        * sequence_length
        * width
        * model_config.d_ff
    )
    original = 2 * original_per_block
    dense = sum(dense_components.values())
    triangular = sum(triangular_components.values())
    return {
        "original_two_block_estimated_multiplies": original,
        "unfused_modal_logical_multiplies": logical_multiplies,
        "fused_dense_executed_multiplies": dense,
        "fused_triangular_nonzero_multiplies": triangular,
        "dense_components": dense_components,
        "triangular_components": triangular_components,
        "fused_dense_vs_original_ratio": dense / original,
        "fused_triangular_vs_original_ratio": triangular / original,
        "fused_dense_vs_unfused_modal_ratio": (
            dense / logical_multiplies
        ),
        "fused_triangular_vs_unfused_modal_ratio": (
            triangular / logical_multiplies
        ),
        "counting_scope": (
            "two replaced blocks only; scalar multiplies; bias, GELU, "
            "embedding, final norm, and vocabulary head excluded"
        ),
        "dense_interpretation": (
            "current einsum fast path includes structural zeros in its "
            "dense kernels"
        ),
        "triangular_interpretation": (
            "packed causal-pair PyTorch reference executes only the "
            "lower-triangular position pairs; wall-clock behavior is "
            "reported in its separate benchmark"
        ),
    }


def _fused_rebuild_bridge(
    stack: FusedTwoLayerModalStack,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = stack.first
    second = stack.second
    first_modes = first.config.output_modes
    second_modes = second.config.input_modes
    coordinate_map = torch.zeros(
        first.sequence_length,
        first_modes,
        first.width,
        dtype=first.input_mean.dtype,
        device=first.input_mean.device,
    )
    coordinate_map[:, :, :first_modes] = torch.eye(
        first_modes,
        dtype=coordinate_map.dtype,
        device=coordinate_map.device,
    )
    coordinate_map[:, :, first_modes:] = first.completion_weight
    coordinate_bias = torch.zeros(
        first.sequence_length,
        first.width,
        dtype=coordinate_map.dtype,
        device=coordinate_map.device,
    )
    coordinate_bias[:, first_modes:] = first.completion_bias
    first_weight = first.output_weight * first.output_scale.unsqueeze(1)
    first_bias = first.output_bias * first.output_scale
    to_second = coordinate_map[:, :, :second_modes]
    second_coordinate_weight = torch.einsum(
        "shk,skq->shq",
        first_weight,
        to_second,
    )
    second_coordinate_bias = (
        torch.einsum("sk,skq->sq", first_bias, to_second)
        + coordinate_bias[:, :second_modes]
    )
    normalized_second_kernel = (
        second.coordinate_kernel
        / second.input_scale.view(
            1,
            second.sequence_length,
            second.config.input_modes,
            1,
        )
    )
    return (
        torch.einsum(
            "shq,tsqr->tshr",
            second_coordinate_weight,
            normalized_second_kernel,
        ),
        second.hidden_bias
        + torch.einsum(
            "sq,tsqr->tr",
            second_coordinate_bias,
            normalized_second_kernel,
        ),
    )


def _verify_fused_causality(
    stack: FusedTwoLayerModalStack,
) -> None:
    sequence_length = stack.config.first.sequence_length
    future = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=stack.first.input_mean.device,
        ),
        diagonal=1,
    )
    causal_kernels = {
        "first.coordinate_kernel": stack.first.coordinate_kernel,
        "first.input_kernel": stack.first.input_kernel,
        "second.coordinate_kernel": stack.second.coordinate_kernel,
        "second.input_kernel": stack.second.input_kernel,
        "bridge_kernel": stack.bridge_kernel,
    }
    for name, kernel in causal_kernels.items():
        if torch.count_nonzero(kernel[future]).item() != 0:
            raise ValueError(
                f"fused causal kernel has a future-position edge: {name}"
            )

    generator = torch.Generator(device="cpu").manual_seed(809)
    probe = (
        stack.first.input_mean.unsqueeze(0).repeat(2, 1, 1)
        + 0.05
        * torch.randn(
            2,
            sequence_length,
            stack.config.first.width,
            generator=generator,
            dtype=stack.first.input_mean.dtype,
            device=stack.first.input_mean.device,
        )
    )
    with torch.no_grad():
        baseline = stack.forward_fast(probe)
        for position in range(sequence_length - 1):
            changed = probe.clone()
            changed[:, position + 1 :] += 0.125
            changed_output = stack.forward_fast(changed)
            torch.testing.assert_close(
                baseline[:, : position + 1],
                changed_output[:, : position + 1],
                rtol=0,
                atol=0,
            )


_LAZY_STATUS_FIELDS = {
    "residency",
    "loaded",
    "last_dispatch",
    "fast_path_calls",
    "instrumented_path_calls",
    "load_attempts",
    "successful_loads",
    "cache_hits",
    "failed_loads",
    "evictions",
    "derived_kernel_verifications",
    "resident_fast_tensor_bytes",
    "resident_sidecar_tensor_bytes",
    "sidecar_file_bytes_read",
    "last_error",
}


def _lazy_status_dict(
    stack: LazyFusedTwoLayerModalStack,
) -> dict[str, object]:
    return asdict(stack.instrumentation_status())


def _validate_lazy_status(
    value: object,
    *,
    path: str,
    require_unloaded_fast_only: bool,
) -> dict[str, object]:
    status = _object(value, path)
    if set(status) != _LAZY_STATUS_FIELDS:
        raise ValueError(f"{path} fields mismatch")
    residency = status.get("residency")
    if residency not in {"unloaded", "loading", "loaded", "failed"}:
        raise ValueError(f"{path}.residency is invalid")
    if not isinstance(status.get("loaded"), bool):
        raise ValueError(f"{path}.loaded must be boolean")
    last_dispatch = status.get("last_dispatch")
    if last_dispatch not in {
        None,
        "fast_cross_layer",
        "logical_sidecar",
    }:
        raise ValueError(f"{path}.last_dispatch is invalid")
    for name in (
        "fast_path_calls",
        "instrumented_path_calls",
        "load_attempts",
        "successful_loads",
        "cache_hits",
        "failed_loads",
        "evictions",
        "derived_kernel_verifications",
        "resident_fast_tensor_bytes",
        "resident_sidecar_tensor_bytes",
        "sidecar_file_bytes_read",
    ):
        if _integer(status.get(name), f"{path}.{name}") < 0:
            raise ValueError(f"{path}.{name} is negative")
    if status.get("last_error") is not None and not isinstance(
        status.get("last_error"),
        str,
    ):
        raise ValueError(f"{path}.last_error must be null or a string")
    if (
        _integer(
            status.get("resident_fast_tensor_bytes"),
            f"{path}.resident_fast_tensor_bytes",
        )
        != 199_808
    ):
        raise ValueError(f"{path} fast-state byte count mismatch")
    if require_unloaded_fast_only:
        for name in (
            "instrumented_path_calls",
            "load_attempts",
            "successful_loads",
            "cache_hits",
            "failed_loads",
            "evictions",
            "derived_kernel_verifications",
            "resident_sidecar_tensor_bytes",
            "sidecar_file_bytes_read",
        ):
            if status[name] != 0:
                raise ValueError(
                    f"{path}.{name} shows sidecar activity"
                )
        if (
            status["residency"] != "unloaded"
            or status["loaded"] is not False
            or status["last_error"] is not None
        ):
            raise ValueError(f"{path} is not an unloaded fast-only state")
    return status


def _fused_trace_contract(
    *,
    validation_inputs: torch.Tensor,
    unfused: ToyTransformer,
    lazy: FusedToyTransformer,
    lazy_stack: LazyFusedTwoLayerModalStack,
    expected_sidecar_file_bytes: int,
) -> dict[str, object]:
    status_before = _lazy_status_dict(lazy_stack)
    _validate_lazy_status(
        status_before,
        path="lazy status before instrumentation",
        require_unloaded_fast_only=True,
    )
    with torch.inference_mode():
        unfused_output = unfused(
            validation_inputs,
            capture_activations=True,
            retain_activation_gradients=False,
        )
        lazy_trace_output = lazy(
            validation_inputs,
            capture_activations=True,
            retain_activation_gradients=False,
        )
    status_after_first_capture = _lazy_status_dict(lazy_stack)
    with torch.inference_mode():
        identity_output = lazy(
            validation_inputs,
            activation_interventions={
                "layer.0.modal.hidden": lambda values: values,
            },
        )
    status_after_reused_intervention = _lazy_status_dict(lazy_stack)
    with torch.inference_mode():
        lazy_fast_output = lazy(validation_inputs)
    status_after_fast_reuse = _lazy_status_dict(lazy_stack)
    eviction_returned_true = lazy_stack.evict_instrumentation()
    status_after_explicit_eviction = _lazy_status_dict(lazy_stack)

    for label, status in (
        ("after first capture", status_after_first_capture),
        ("after reused intervention", status_after_reused_intervention),
        ("after fast reuse", status_after_fast_reuse),
        ("after explicit eviction", status_after_explicit_eviction),
    ):
        _validate_lazy_status(
            status,
            path=f"lazy status {label}",
            require_unloaded_fast_only=False,
        )
    first = status_after_first_capture
    if (
        first["residency"] != "loaded"
        or first["loaded"] is not True
        or first["last_dispatch"] != "logical_sidecar"
        or first["load_attempts"] != 1
        or first["successful_loads"] != 1
        or first["failed_loads"] != 0
        or first["instrumented_path_calls"] != 1
        or first["cache_hits"] != 0
        or first["derived_kernel_verifications"] != 1
        or first["resident_sidecar_tensor_bytes"] != 203_648
        or first["sidecar_file_bytes_read"]
        != expected_sidecar_file_bytes
    ):
        raise ValueError(
            "first lazy instrumentation capture did not load exactly once"
        )
    reused = status_after_reused_intervention
    if (
        reused["load_attempts"] != 1
        or reused["successful_loads"] != 1
        or reused["cache_hits"] != 1
        or reused["instrumented_path_calls"] != 2
        or reused["derived_kernel_verifications"] != 1
        or reused["last_dispatch"] != "logical_sidecar"
    ):
        raise ValueError("lazy intervention did not reuse instrumentation")
    if (
        status_after_fast_reuse["last_dispatch"]
        != "fast_cross_layer"
        or status_after_fast_reuse["fast_path_calls"] != 1
        or status_after_fast_reuse["successful_loads"] != 1
    ):
        raise ValueError(
            "loaded lazy runtime did not retain its fast dispatch"
        )
    if (
        not eviction_returned_true
        or status_after_explicit_eviction["residency"] != "unloaded"
        or status_after_explicit_eviction["loaded"] is not False
        or status_after_explicit_eviction[
            "resident_sidecar_tensor_bytes"
        ]
        != 0
        or status_after_explicit_eviction["evictions"] != 1
        or status_after_explicit_eviction["load_attempts"] != 1
        or status_after_explicit_eviction["successful_loads"] != 1
        or status_after_explicit_eviction[
            "derived_kernel_verifications"
        ]
        != 1
    ):
        raise ValueError(
            "lazy instrumentation eviction did not release state"
        )
    if (
        unfused_output.activations is None
        or lazy_trace_output.activations is None
    ):
        raise ValueError("lazy trace verification captured no activations")
    unfused_trace = unfused_output.activations
    lazy_trace = lazy_trace_output.activations
    if lazy_trace.names != unfused_trace.names:
        raise ValueError("lazy logical trace names differ from unfused")
    maximum_by_tap = {
        name: (
            lazy_trace[name] - unfused_trace[name]
        ).abs().max().item()
        for name in unfused_trace.names
    }
    if any(value != 0.0 for value in maximum_by_tap.values()):
        raise ValueError("lazy logical trace tensors differ from unfused")
    trace_logit_difference = (
        lazy_trace_output.logits - unfused_output.logits
    ).abs().max().item()
    if trace_logit_difference != 0.0:
        raise ValueError("lazy logical trace logits differ from unfused")
    identity_difference = (
        identity_output.logits - lazy_trace_output.logits
    ).abs().max().item()
    if identity_difference != 0.0:
        raise ValueError("lazy identity intervention changed output")
    return {
        "default_dispatch": (
            "seven-tensor forward_fast with exact cross-layer modal bypass"
        ),
        "trace_dispatch": (
            "load the verified four-artifact logical sidecar on first "
            "capture or intervention, then reuse it"
        ),
        "fast_path_has_no_activation_trace": (
            lazy_fast_output.activations is None
        ),
        "capture_returns_activation_trace": True,
        "trace_names_exactly_equal_to_unfused": True,
        "trace_names": list(lazy_trace.names),
        "maximum_unfused_to_fused_difference_by_trace_tap": (
            maximum_by_tap
        ),
        "maximum_unfused_to_fused_trace_logit_difference": (
            trace_logit_difference
        ),
        "maximum_fast_to_trace_logit_difference": (
            (
                lazy_fast_output.logits - lazy_trace_output.logits
            ).abs().max().item()
        ),
        "identity_intervention_tap": "layer.0.modal.hidden",
        "identity_intervention_applied": True,
        "identity_intervention_maximum_logit_difference": (
            identity_difference
        ),
        "status_before_instrumentation": status_before,
        "status_after_first_capture": status_after_first_capture,
        "status_after_reused_intervention": (
            status_after_reused_intervention
        ),
        "status_after_fast_reuse": status_after_fast_reuse,
        "explicit_eviction_returned_true": eviction_returned_true,
        "status_after_explicit_eviction": (
            status_after_explicit_eviction
        ),
        "sidecar_loaded_exactly_once": True,
        "repeated_instrumentation_reused_cache": True,
        "fast_dispatch_remained_available_while_sidecar_loaded": True,
        "explicit_eviction_released_sidecar_tensors": True,
    }


def _fused_percentile(
    ordered: tuple[float, ...],
    fraction: float,
) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def _verify_fused_benchmark(
    environment_value: object,
    benchmark_value: object,
    *,
    systems: tuple[str, ...] = (
        "teacher",
        "unfused",
        "monolithic",
        "lazy",
    ),
    speedup_pairs: Mapping[str, tuple[str, str]] | None = None,
    minimum_speedup_name: str = "lazy_vs_unfused",
) -> tuple[int, float]:
    if speedup_pairs is None:
        speedup_pairs = {
            "monolithic_vs_unfused": ("unfused", "monolithic"),
            "lazy_vs_unfused": ("unfused", "lazy"),
            "monolithic_vs_teacher": ("teacher", "monolithic"),
            "lazy_vs_teacher": ("teacher", "lazy"),
            "lazy_vs_monolithic": ("monolithic", "lazy"),
        }
    if (
        not systems
        or len(set(systems)) != len(systems)
        or any(not isinstance(name, str) or not name for name in systems)
    ):
        raise ValueError("fused benchmark verifier systems are invalid")
    if minimum_speedup_name not in speedup_pairs:
        raise ValueError("fused benchmark minimum speedup is undeclared")
    if any(
        not isinstance(name, str)
        or not name
        or len(pair) != 2
        or pair[0] not in systems
        or pair[1] not in systems
        or pair[0] == pair[1]
        for name, pair in speedup_pairs.items()
    ):
        raise ValueError("fused benchmark verifier speedup pairs are invalid")

    environment = _object(
        environment_value,
        "fused benchmark_environment",
    )
    expected_environment_fields = {
        "python_version",
        "torch_version",
        "platform",
        "machine",
        "processor",
        "device",
        "dtype",
        "torch_intraop_threads_outside_benchmark",
        "torch_interop_threads",
        "benchmark_contract",
    }
    if set(environment) != expected_environment_fields:
        raise ValueError("fused benchmark environment fields mismatch")
    for name in (
        "python_version",
        "torch_version",
        "platform",
        "machine",
        "processor",
    ):
        value = environment.get(name)
        if not isinstance(value, str):
            raise ValueError(
                f"fused benchmark environment {name} must be a string"
            )
    if environment.get("device") != "cpu":
        raise ValueError("fused benchmark device mismatch")
    if environment.get("dtype") != "float32":
        raise ValueError("fused benchmark dtype mismatch")
    for name in (
        "torch_intraop_threads_outside_benchmark",
        "torch_interop_threads",
    ):
        if _integer(
            environment.get(name),
            f"fused benchmark environment.{name}",
        ) <= 0:
            raise ValueError(
                f"fused benchmark environment.{name} must be positive"
            )

    contract = _object(
        environment.get("benchmark_contract"),
        "fused benchmark contract",
    )
    expected_contract = {
        "input_split": "validation_fisher",
        "batch_sizes": [1, 8, 64, 256],
        "systems": list(systems),
        "intraop_threads": 1,
        "inference_mode": True,
        "repeats": 9,
        "minimum_block_seconds": 0.2,
        "warmup_iterations": 100,
        "minimum_warmup_seconds": 1.0,
        "ordering": "deterministic rotating system order",
        "scope": (
            "end-to-end fixed-length model forward including embedding, "
            "compiled blocks, final norm, and vocabulary head"
        ),
    }
    if contract != expected_contract:
        raise ValueError("fused benchmark contract mismatch")

    benchmark = _array(benchmark_value, "fused benchmark")
    if len(benchmark) != len(expected_contract["batch_sizes"]):
        raise ValueError("fused benchmark batch count mismatch")
    minimum_observed_speedup = math.inf
    for batch_index, (value, expected_batch) in enumerate(
        zip(
            benchmark,
            expected_contract["batch_sizes"],
            strict=True,
        )
    ):
        path = f"fused benchmark[{batch_index}]"
        batch = _object(value, path)
        expected_fields = {
            "batch_size",
            "timings",
            "examples_per_second",
            "speedup_ratios",
            "round_orders",
        }
        if set(batch) != expected_fields:
            raise ValueError(f"{path} fields mismatch")
        if _integer(
            batch.get("batch_size"),
            f"{path}.batch_size",
        ) != expected_batch:
            raise ValueError(f"{path}.batch_size mismatch")
        timings = _object(batch.get("timings"), f"{path}.timings")
        if set(timings) != set(systems):
            raise ValueError(f"{path}.timings systems mismatch")
        medians: dict[str, float] = {}
        for system in systems:
            timing_path = f"{path}.timings.{system}"
            timing = _object(timings.get(system), timing_path)
            expected_timing_fields = {
                "repeats",
                "iterations_per_block",
                "target_block_seconds",
                "calibration_seconds",
                "warmup_calls",
                "raw_microseconds",
                "median_microseconds",
                "minimum_microseconds",
                "maximum_microseconds",
                "p10_microseconds",
                "p90_microseconds",
            }
            if set(timing) != expected_timing_fields:
                raise ValueError(f"{timing_path} fields mismatch")
            repeats = _integer(
                timing.get("repeats"),
                f"{timing_path}.repeats",
            )
            iterations = _integer(
                timing.get("iterations_per_block"),
                f"{timing_path}.iterations_per_block",
            )
            warmup_calls = _integer(
                timing.get("warmup_calls"),
                f"{timing_path}.warmup_calls",
            )
            target_seconds = _number(
                timing.get("target_block_seconds"),
                f"{timing_path}.target_block_seconds",
            )
            calibration_seconds = _number(
                timing.get("calibration_seconds"),
                f"{timing_path}.calibration_seconds",
            )
            if (
                repeats != expected_contract["repeats"]
                or iterations <= 0
                or iterations & (iterations - 1)
                or warmup_calls
                < expected_contract["warmup_iterations"]
                or target_seconds
                != expected_contract["minimum_block_seconds"]
                or calibration_seconds < target_seconds
            ):
                raise ValueError(f"{timing_path} timing contract mismatch")
            raw = tuple(
                _number(item, f"{timing_path}.raw_microseconds")
                for item in _array(
                    timing.get("raw_microseconds"),
                    f"{timing_path}.raw_microseconds",
                )
            )
            if len(raw) != repeats or any(item <= 0 for item in raw):
                raise ValueError(
                    f"{timing_path}.raw_microseconds is invalid"
                )
            ordered = tuple(sorted(raw))
            expected_summaries = {
                "median_microseconds": statistics.median(raw),
                "minimum_microseconds": ordered[0],
                "maximum_microseconds": ordered[-1],
                "p10_microseconds": _fused_percentile(ordered, 0.1),
                "p90_microseconds": _fused_percentile(ordered, 0.9),
            }
            for name, expected in expected_summaries.items():
                actual = _number(
                    timing.get(name),
                    f"{timing_path}.{name}",
                )
                if actual <= 0 or not math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{timing_path}.{name} is inconsistent"
                    )
            medians[system] = expected_summaries[
                "median_microseconds"
            ]

        examples = _object(
            batch.get("examples_per_second"),
            f"{path}.examples_per_second",
        )
        if set(examples) != set(systems):
            raise ValueError(
                f"{path}.examples_per_second systems mismatch"
            )
        for system in systems:
            actual = _number(
                examples.get(system),
                f"{path}.examples_per_second.{system}",
            )
            expected = expected_batch * 1e6 / medians[system]
            if actual <= 0 or not math.isclose(
                actual,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{path}.examples_per_second.{system} mismatch"
                )

        speedups = _object(
            batch.get("speedup_ratios"),
            f"{path}.speedup_ratios",
        )
        expected_speedups = {
            name: medians[reference] / medians[candidate]
            for name, (reference, candidate) in speedup_pairs.items()
        }
        if set(speedups) != set(expected_speedups):
            raise ValueError(f"{path}.speedup_ratios fields mismatch")
        for name, expected in expected_speedups.items():
            actual = _number(
                speedups.get(name),
                f"{path}.speedup_ratios.{name}",
            )
            if actual <= 0 or not math.isclose(
                actual,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{path}.speedup_ratios.{name} mismatch"
                )
        minimum_observed_speedup = min(
            minimum_observed_speedup,
            expected_speedups[minimum_speedup_name],
        )

        orders = _array(
            batch.get("round_orders"),
            f"{path}.round_orders",
        )
        expected_orders = [
            list(systems[offset:] + systems[:offset])
            for repeat in range(expected_contract["repeats"])
            for offset in (repeat % len(systems),)
        ]
        if orders != expected_orders:
            raise ValueError(f"{path}.round_orders mismatch")
    return len(benchmark), minimum_observed_speedup


def _fused_lazy_benchmark_comparison(
    benchmark_value: object,
) -> dict[str, object]:
    benchmark = _array(
        benchmark_value,
        "fused benchmark for lazy comparison",
    )
    per_batch: list[dict[str, object]] = []
    ratios: list[float] = []
    for index, value in enumerate(benchmark):
        batch = _object(value, f"fused benchmark[{index}]")
        timings = _object(
            batch.get("timings"),
            f"fused benchmark[{index}].timings",
        )
        speedups = _object(
            batch.get("speedup_ratios"),
            f"fused benchmark[{index}].speedup_ratios",
        )
        monolithic = _number(
            _object(
                timings.get("monolithic"),
                f"fused benchmark[{index}].timings.monolithic",
            ).get("median_microseconds"),
            f"fused benchmark[{index}].monolithic median",
        )
        lazy = _number(
            _object(
                timings.get("lazy"),
                f"fused benchmark[{index}].timings.lazy",
            ).get("median_microseconds"),
            f"fused benchmark[{index}].lazy median",
        )
        ratio = lazy / monolithic
        ratios.append(ratio)
        per_batch.append(
            {
                "batch_size": _integer(
                    batch.get("batch_size"),
                    f"fused benchmark[{index}].batch_size",
                ),
                "lazy_to_monolithic_latency_ratio": ratio,
                "lazy_vs_monolithic_speedup": _number(
                    speedups.get("lazy_vs_monolithic"),
                    f"fused benchmark[{index}].lazy_vs_monolithic",
                ),
                "lazy_latency_regression_fraction": ratio - 1.0,
            }
        )
    if not ratios:
        raise ValueError("fused lazy benchmark comparison is empty")
    geometric_mean = math.exp(
        sum(math.log(value) for value in ratios) / len(ratios)
    )
    maximum = max(ratios)
    return {
        "per_batch": per_batch,
        "geometric_mean_lazy_to_monolithic_latency_ratio": (
            geometric_mean
        ),
        "geometric_mean_lazy_latency_regression_fraction": (
            geometric_mean - 1.0
        ),
        "maximum_lazy_to_monolithic_latency_ratio": maximum,
        "maximum_lazy_latency_regression_fraction": maximum - 1.0,
        "hard_latency_gate_applied": False,
        "interpretation": (
            "positive regression fractions mean the lazy fast wrapper was "
            "slower; negative values mean it was faster"
        ),
    }


def _fused_triangular_benchmark_comparison(
    benchmark_value: object,
) -> dict[str, object]:
    """Recompute every reported packed-triangular benchmark comparison."""

    benchmark = _array(
        benchmark_value,
        "fused triangular benchmark for comparison",
    )
    per_batch: list[dict[str, object]] = []
    lazy_speedups: list[float] = []
    unfused_speedups: list[float] = []
    for index, value in enumerate(benchmark):
        path = f"fused triangular benchmark[{index}]"
        batch = _object(value, path)
        timings = _object(batch.get("timings"), f"{path}.timings")
        speedups = _object(
            batch.get("speedup_ratios"),
            f"{path}.speedup_ratios",
        )
        lazy_median = _number(
            _object(
                timings.get("lazy"),
                f"{path}.timings.lazy",
            ).get("median_microseconds"),
            f"{path}.lazy median",
        )
        unfused_median = _number(
            _object(
                timings.get("unfused"),
                f"{path}.timings.unfused",
            ).get("median_microseconds"),
            f"{path}.unfused median",
        )
        triangular_median = _number(
            _object(
                timings.get("triangular"),
                f"{path}.timings.triangular",
            ).get("median_microseconds"),
            f"{path}.triangular median",
        )
        lazy_speedup = _number(
            speedups.get("triangular_vs_lazy"),
            f"{path}.triangular_vs_lazy",
        )
        unfused_speedup = _number(
            speedups.get("triangular_vs_unfused"),
            f"{path}.triangular_vs_unfused",
        )
        lazy_speedups.append(lazy_speedup)
        unfused_speedups.append(unfused_speedup)
        per_batch.append(
            {
                "batch_size": _integer(
                    batch.get("batch_size"),
                    f"{path}.batch_size",
                ),
                "triangular_vs_lazy_speedup": lazy_speedup,
                "triangular_to_lazy_latency_ratio": (
                    triangular_median / lazy_median
                ),
                "triangular_vs_unfused_speedup": unfused_speedup,
                "triangular_to_unfused_latency_ratio": (
                    triangular_median / unfused_median
                ),
            }
        )
    if not lazy_speedups:
        raise ValueError("fused triangular benchmark comparison is empty")
    return {
        "per_batch": per_batch,
        "geometric_mean_triangular_vs_lazy_speedup": math.exp(
            sum(math.log(value) for value in lazy_speedups)
            / len(lazy_speedups)
        ),
        "geometric_mean_triangular_vs_unfused_speedup": math.exp(
            sum(math.log(value) for value in unfused_speedups)
            / len(unfused_speedups)
        ),
        "hard_latency_gate_applied": False,
        "interpretation": (
            "speedups above one mean the packed triangular runtime was faster; "
            "latency ratios below one mean it was faster"
        ),
    }


def _verify_fused_triangular_runtime(
    section_value: object,
    *,
    source_lazy_path: Path,
    source_lazy_sha256: str,
    lazy_stack: LazyFusedTwoLayerModalStack,
    lazy_model: FusedToyTransformer,
    teacher: ToyTransformer,
    validation_split,
    gate: Mapping[str, float],
    arithmetic: Mapping[str, object],
) -> dict[str, object]:
    """Verify the optional v3 packed runtime without rerunning wall time."""

    section = _object(
        section_value,
        "fused executor triangular_runtime_benchmark",
    )
    expected_section_fields = {
        "source_lazy_artifact",
        "runtime_contract",
        "source_lazy_status_before",
        "source_lazy_status_after",
        "validation",
        "benchmark_environment",
        "benchmark",
        "comparison",
    }
    if set(section) != expected_section_fields:
        raise ValueError(
            "fused triangular runtime benchmark fields mismatch"
        )

    expected_source = {
        "filename": source_lazy_path.name,
        "sha256": source_lazy_sha256,
        "artifact_kind": "lazy_fused_two_layer_modal_stack",
        "format_version": 2,
    }
    _assert_numeric_tree_match(
        section.get("source_lazy_artifact"),
        expected_source,
        path="fused triangular source_lazy_artifact",
    )

    source_state_before = _module_state_sha256(lazy_stack)
    actual_source_status_before = _lazy_status_dict(lazy_stack)
    _validate_lazy_status(
        actual_source_status_before,
        path="fused triangular verifier source status before derivation",
        require_unloaded_fast_only=True,
    )
    triangular_stack = (
        PackedTriangularFusedTwoLayerModalStack.from_lazy(lazy_stack)
    )
    actual_source_status_after_derivation = _lazy_status_dict(lazy_stack)
    _assert_numeric_tree_match(
        actual_source_status_after_derivation,
        actual_source_status_before,
        path="fused triangular source status across derivation",
    )
    if _module_state_sha256(lazy_stack) != source_state_before:
        raise ValueError(
            "packed triangular derivation mutated the lazy source runtime"
        )

    expected_pair_count = (
        lazy_stack.sequence_length * (lazy_stack.sequence_length + 1) // 2
    )
    if triangular_stack.causal_pair_count != expected_pair_count:
        raise ValueError("packed triangular causal-pair count mismatch")
    expected_target, expected_source_indices = torch.tril_indices(
        lazy_stack.sequence_length,
        lazy_stack.sequence_length,
        device=lazy_stack.reference_tensor.device,
    )
    if not torch.equal(
        triangular_stack.causal_target_indices,
        expected_target,
    ) or not torch.equal(
        triangular_stack.causal_source_indices,
        expected_source_indices,
    ):
        raise ValueError("packed triangular causal index order mismatch")
    expected_packed_tensors = {
        "first_input_mean": lazy_stack.first_input_mean,
        "first_hidden_bias": lazy_stack.first_hidden_bias,
        "bridge_bias": lazy_stack.bridge_bias,
        "second_fused_output_weight": (
            lazy_stack.second_fused_output_weight
        ),
        "second_fused_output_bias": lazy_stack.second_fused_output_bias,
        "packed_first_input_kernel": lazy_stack.first_input_kernel[
            expected_target,
            expected_source_indices,
        ],
        "packed_bridge_kernel": lazy_stack.bridge_kernel[
            expected_target,
            expected_source_indices,
        ],
        "causal_target_indices": expected_target,
        "causal_source_indices": expected_source_indices,
    }
    if set(triangular_stack.state_dict()) != set(expected_packed_tensors):
        raise ValueError("packed triangular state fields mismatch")
    for name, expected in expected_packed_tensors.items():
        if not torch.equal(
            triangular_stack.state_dict()[name],
            expected,
        ):
            raise ValueError(
                f"packed triangular state is not source-derived: {name}"
            )
    triangular_storage = _fused_state_storage(triangular_stack)
    if (
        triangular_storage["total_state_bytes"]
        != triangular_stack.packed_state_bytes
        or list(triangular_stack.parameters())
    ):
        raise ValueError("packed triangular state accounting mismatch")
    if any(
        isinstance(module, TransformerBlock)
        for module in triangular_stack.modules()
    ):
        raise ValueError("packed triangular runtime contains a transformer block")
    provenance = triangular_stack.runtime_provenance()
    if provenance.get("source_provenance") != lazy_stack.provenance:
        raise ValueError("packed triangular source provenance mismatch")
    if (
        provenance.get("source_fast_state_bytes") != lazy_stack.fast_state_bytes
        or provenance.get("dense_scalar_multiplies")
        != arithmetic["fused_dense_executed_multiplies"]
        or provenance.get("packed_scalar_multiplies")
        != arithmetic["fused_triangular_nonzero_multiplies"]
    ):
        raise ValueError("packed triangular derived facts mismatch")

    expected_runtime_contract = {
        "implementation": "packed_triangular_prefix_v1",
        "serialized_artifact": False,
        "default_backend": False,
        "weights_updated": False,
        "test_used": False,
        "validation_split": "validation_fisher",
        "benchmark_split": "validation_fisher",
        "packed_causal_pair_count": expected_pair_count,
        "packed_fast_state_tensor_bytes": (
            triangular_stack.packed_state_bytes
        ),
    }
    _assert_numeric_tree_match(
        section.get("runtime_contract"),
        expected_runtime_contract,
        path="fused triangular runtime_contract",
    )

    reported_before = _validate_lazy_status(
        section.get("source_lazy_status_before"),
        path="fused triangular reported source status before",
        require_unloaded_fast_only=True,
    )
    reported_after = _validate_lazy_status(
        section.get("source_lazy_status_after"),
        path="fused triangular reported source status after",
        require_unloaded_fast_only=True,
    )
    if (
        reported_after["fast_path_calls"]
        <= reported_before["fast_path_calls"]
        or reported_after["last_dispatch"] != "fast_cross_layer"
    ):
        raise ValueError(
            "fused triangular source statuses do not prove fast-only use"
        )

    triangular_model = FusedToyTransformer.from_teacher(
        teacher,
        triangular_stack,
    )
    _fused_freeze(triangular_model)
    if list(triangular_model.parameters()):
        raise ValueError(
            "packed triangular full runtime unexpectedly contains parameters"
        )
    lazy_logits = associative_recall_answer_logits(
        lazy_model,
        validation_split,
    )
    triangular_logits = associative_recall_answer_logits(
        triangular_model,
        validation_split,
    )
    recomputed_comparison = _fused_comparison(
        split=validation_split,
        unfused_logits=lazy_logits,
        fused_logits=triangular_logits,
    )
    validation = _object(
        section.get("validation"),
        "fused triangular validation",
    )
    if set(validation) != {"gate", "gate_passed", "triangular_vs_lazy"}:
        raise ValueError("fused triangular validation fields mismatch")
    _assert_numeric_tree_match(
        validation.get("gate"),
        dict(gate),
        path="fused triangular validation.gate",
    )
    _assert_numeric_tree_match(
        validation.get("triangular_vs_lazy"),
        recomputed_comparison,
        path="fused triangular validation.triangular_vs_lazy",
    )
    gate_passed = _fused_gate_passed(recomputed_comparison, gate)
    if validation.get("gate_passed") is not gate_passed or not gate_passed:
        raise ValueError("fused triangular validation gate failed")

    actual_source_status_after_validation = _lazy_status_dict(lazy_stack)
    _validate_lazy_status(
        actual_source_status_after_validation,
        path="fused triangular verifier source status after validation",
        require_unloaded_fast_only=True,
    )
    if (
        actual_source_status_after_validation["fast_path_calls"]
        <= actual_source_status_after_derivation["fast_path_calls"]
        or actual_source_status_after_validation["last_dispatch"]
        != "fast_cross_layer"
    ):
        raise ValueError(
            "fused triangular validation did not use the lazy fast reference"
        )
    if (
        _module_state_sha256(lazy_stack) != source_state_before
        or _sha256(source_lazy_path) != source_lazy_sha256
    ):
        raise ValueError(
            "fused triangular verification changed its lazy source"
        )

    triangular_systems = (
        "teacher",
        "unfused",
        "monolithic",
        "lazy",
        "triangular",
    )
    triangular_speedup_pairs = {
        "monolithic_vs_unfused": ("unfused", "monolithic"),
        "lazy_vs_unfused": ("unfused", "lazy"),
        "monolithic_vs_teacher": ("teacher", "monolithic"),
        "lazy_vs_teacher": ("teacher", "lazy"),
        "lazy_vs_monolithic": ("monolithic", "lazy"),
        "triangular_vs_unfused": ("unfused", "triangular"),
        "triangular_vs_teacher": ("teacher", "triangular"),
        "triangular_vs_monolithic": ("monolithic", "triangular"),
        "triangular_vs_lazy": ("lazy", "triangular"),
    }
    benchmark_batch_count, minimum_speedup = _verify_fused_benchmark(
        section.get("benchmark_environment"),
        section.get("benchmark"),
        systems=triangular_systems,
        speedup_pairs=triangular_speedup_pairs,
        minimum_speedup_name="triangular_vs_lazy",
    )
    comparison = _fused_triangular_benchmark_comparison(
        section.get("benchmark")
    )
    _assert_numeric_tree_match(
        section.get("comparison"),
        comparison,
        path="fused triangular comparison",
    )

    return {
        "triangular_runtime_present": True,
        "triangular_implementation": "packed_triangular_prefix_v1",
        "triangular_serialized_artifact": False,
        "triangular_default_backend": False,
        "triangular_causal_pair_count": expected_pair_count,
        "triangular_fast_stack_resident_tensor_bytes": (
            triangular_stack.packed_state_bytes
        ),
        "triangular_validation_gate_passed": True,
        "triangular_zero_source_sidecar_loads": True,
        "triangular_benchmark_batch_count": benchmark_batch_count,
        "minimum_observed_triangular_vs_lazy_speedup": minimum_speedup,
        "geometric_mean_triangular_vs_lazy_speedup": comparison[
            "geometric_mean_triangular_vs_lazy_speedup"
        ],
        "geometric_mean_triangular_vs_unfused_speedup": comparison[
            "geometric_mean_triangular_vs_unfused_speedup"
        ],
        "triangular_benchmark_hard_latency_gate_applied": False,
    }


def _verify_optional_fused_executor(
    directory: Path,
    *,
    checkpoint_hash: str,
    fisher_path: Path,
    bases: Mapping[str, FisherModeBasis],
    model: ToyTransformer,
    model_config: TransformerConfig,
    splits: AssociativeRecallSplits,
    sequence_length: int,
) -> dict[str, object]:
    paths = fused_executor_artifact_paths(directory)
    bundle = (
        paths.stack,
        paths.runtime,
        paths.report_json,
        paths.report_markdown,
    )
    present = [path.is_file() for path in bundle]
    if not any(present):
        return {"present": False}
    if not all(present):
        missing = [
            path.name
            for path, exists in zip(bundle, present, strict=True)
            if not exists
        ]
        raise FileNotFoundError(
            f"incomplete fused executor artifacts: {missing}"
        )
    if len(model.layers) != 2:
        raise ValueError(
            "fused executor report requires exactly two model layers"
        )

    report = _object(
        json.loads(paths.report_json.read_text()),
        "fused executor report",
    )
    _assert_finite_tree(report, "fused executor report")
    report_format_version = _integer(
        report.get("format_version"),
        "fused executor format_version",
    )
    if report_format_version not in (2, 3):
        raise ValueError("unsupported fused executor report format")
    expected_report_fields = {
        "format_version",
        "checkpoint_sha256",
        "fisher_sha256",
        "split_manifest_sha256",
        "teacher_state_sha256_before",
        "teacher_state_sha256_after",
        "protocol",
        "validation",
        "test",
        "arithmetic",
        "dispatch_and_trace_contract",
        "benchmark_environment",
        "benchmark",
        "lazy_vs_monolithic_benchmark",
        "lazy_fast_runtime_status",
        "storage",
        "source_artifacts",
        "fused_artifact",
        "lazy_fused_artifact",
        "artifacts_locked_before_validation_and_test",
        "scientific_status",
        "elapsed_seconds",
    }
    if report_format_version == 3:
        expected_report_fields.add("triangular_runtime_benchmark")
    if set(report) != expected_report_fields:
        raise ValueError("fused executor report fields mismatch")
    fisher_hash = _sha256(fisher_path)
    manifest_path = directory / "split_manifest.json"
    manifest_hash = _sha256(manifest_path)
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("fused executor checkpoint hash mismatch")
    if report.get("fisher_sha256") != fisher_hash:
        raise ValueError("fused executor Fisher hash mismatch")
    if report.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("fused executor split-manifest hash mismatch")
    if report.get("artifacts_locked_before_validation_and_test") is not True:
        raise ValueError("fused executor artifacts were not locked")
    if (
        report.get("scientific_status")
        != (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        )
    ):
        raise ValueError("fused executor scientific status mismatch")
    if _number(
        report.get("elapsed_seconds"),
        "fused executor elapsed_seconds",
    ) < 0:
        raise ValueError("fused executor elapsed_seconds is negative")
    if not paths.report_markdown.read_text().strip():
        raise ValueError("fused executor Markdown report is empty")

    teacher_state_hash = _module_state_sha256(model)
    if (
        report.get("teacher_state_sha256_before")
        != teacher_state_hash
        or report.get("teacher_state_sha256_after")
        != teacher_state_hash
    ):
        raise ValueError("fused executor teacher state hash mismatch")
    source_paths = {
        "checkpoint": directory / "checkpoint.pt",
        "fisher_modes": fisher_path,
        "split_manifest": manifest_path,
        "layer_0_executor": modal_executor_artifact_paths(
            directory,
            0,
        ).executor,
        "layer_0_output_completion": (
            modal_completion_artifact_paths(
                directory,
                0,
            ).output_completion
        ),
        "layer_1_executor": modal_executor_artifact_paths(
            directory,
            1,
        ).executor,
        "layer_1_output_completion": (
            modal_completion_artifact_paths(
                directory,
                1,
            ).output_completion
        ),
    }
    missing_sources = [
        path.name for path in source_paths.values() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"incomplete fused source artifacts: {missing_sources}"
        )
    source_hashes_before = {
        name: _sha256(path) for name, path in source_paths.items()
    }
    expected_sources = {
        name: {
            "filename": path.name,
            "sha256": source_hashes_before[name],
        }
        for name, path in source_paths.items()
    }
    _assert_numeric_tree_match(
        report.get("source_artifacts"),
        expected_sources,
        path="fused executor source_artifacts",
    )
    sidecar_source_paths = {
        name: source_paths[name]
        for name in (
            "layer_0_executor",
            "layer_0_output_completion",
            "layer_1_executor",
            "layer_1_output_completion",
        )
    }
    expected_sidecar_descriptors = {
        name: {
            "filename": path.name,
            "sha256": source_hashes_before[name],
            "size_bytes": path.stat().st_size,
        }
        for name, path in sidecar_source_paths.items()
    }
    expected_sidecar_file_bytes = sum(
        path.stat().st_size for path in sidecar_source_paths.values()
    )

    fit_context_ids_sha256 = _tensor_sha256(
        splits.train.context_ids
    )
    selection_context_ids_sha256 = _tensor_sha256(
        splits.validation.context_ids
    )
    layer_0, completed_0, provenance_0 = _load_composition_layer(
        layer_index=0,
        executor_path=source_paths["layer_0_executor"],
        output_completion_path=source_paths[
            "layer_0_output_completion"
        ],
        checkpoint_hash=checkpoint_hash,
        fisher_hash=fisher_hash,
        teacher_state_hash=teacher_state_hash,
        bases=bases,
        input_activation="layer.0.input",
        output_activation="layer.0.output",
        sequence_length=sequence_length,
        width=model_config.d_model,
        fit_context_ids_sha256=fit_context_ids_sha256,
        selection_context_ids_sha256=(
            selection_context_ids_sha256
        ),
    )
    layer_1, completed_1, provenance_1 = _load_composition_layer(
        layer_index=1,
        executor_path=source_paths["layer_1_executor"],
        output_completion_path=source_paths[
            "layer_1_output_completion"
        ],
        checkpoint_hash=checkpoint_hash,
        fisher_hash=fisher_hash,
        teacher_state_hash=teacher_state_hash,
        bases=bases,
        input_activation="layer.0.output",
        output_activation="layer.1.output",
        sequence_length=sequence_length,
        width=model_config.d_model,
        fit_context_ids_sha256=fit_context_ids_sha256,
        selection_context_ids_sha256=(
            selection_context_ids_sha256
        ),
    )

    raw_artifact = torch.load(
        paths.stack,
        map_location="cpu",
        weights_only=True,
    )
    raw = _object(raw_artifact, "fused stack artifact")
    expected_payload_fields = {
        "format_version",
        "artifact_kind",
        "config",
        "state_dict",
        "metadata",
    }
    if set(raw) != expected_payload_fields:
        raise ValueError("fused stack artifact fields mismatch")
    if _integer(
        raw.get("format_version"),
        "fused stack format_version",
    ) != 1:
        raise ValueError("unsupported fused stack artifact format")
    if raw.get("artifact_kind") != "fused_two_layer_modal_stack":
        raise ValueError("fused stack artifact kind mismatch")
    raw_state = _object(
        raw.get("state_dict"),
        "fused stack state_dict",
    )
    for name, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"fused stack state is not a tensor: {name}"
            )
        if not value.is_floating_point():
            raise ValueError(
                f"fused stack state is not floating point: {name}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"fused stack state is nonfinite: {name}")

    stack, config, metadata = load_fused_modal_stack(paths.stack)
    if set(raw_state) != set(stack.state_dict()):
        raise ValueError("fused stack state fields mismatch")
    for name, value in stack.state_dict().items():
        if not torch.equal(value.cpu(), raw_state[name]):
            raise ValueError(
                f"fused stack loader changed state tensor: {name}"
            )
    if _json_normalized(raw.get("config")) != _json_normalized(
        asdict(config)
    ):
        raise ValueError("fused stack raw config mismatch")
    if _json_normalized(raw.get("metadata")) != _json_normalized(
        metadata
    ):
        raise ValueError("fused stack raw metadata mismatch")
    if not stack.uses_cross_layer_bypass:
        raise ValueError("fused stack did not select the exact bypass")
    if config.first.input_activation != "layer.0.input":
        raise ValueError("fused layer 0 input activation mismatch")
    if config.first.output_activation != "layer.0.output":
        raise ValueError("fused layer 0 output activation mismatch")
    if config.second.input_activation != "layer.0.output":
        raise ValueError("fused layer 1 input activation mismatch")
    if config.second.output_activation != "layer.1.output":
        raise ValueError("fused layer 1 output activation mismatch")
    for label, fused_config, source in (
        ("layer 0", config.first, completed_0),
        ("layer 1", config.second, completed_1),
    ):
        base = source.base_executor
        expected_config = {
            "sequence_length": sequence_length,
            "width": model_config.d_model,
            "input_modes": base.input_projection.modes,
            "routing_width": base.graph.hidden_modes,
            "output_modes": base.output_projection.modes,
            "completion_kind": source.output_completion.graph.graph_kind,
        }
        for name, expected in expected_config.items():
            if getattr(fused_config, name) != expected:
                raise ValueError(
                    f"fused {label} config.{name} mismatch"
                )

    if not torch.equal(
        stack.first.output_mean,
        stack.second.input_mean,
    ):
        raise ValueError("fused boundary position means are not exact")
    if not torch.equal(
        stack.first.output_vectors[
            :, : stack.second.config.input_modes
        ],
        stack.second.input_vectors,
    ):
        raise ValueError("fused boundary modal bases are not exact")
    rebuilt_bridge, rebuilt_bias = _fused_rebuild_bridge(stack)
    if not torch.equal(rebuilt_bridge, stack.bridge_kernel):
        raise ValueError("fused bridge kernel is not algebraically derived")
    if not torch.equal(rebuilt_bias, stack.bridge_bias):
        raise ValueError("fused bridge bias is not algebraically derived")
    _verify_fused_causality(stack)
    if list(stack.parameters()):
        raise ValueError("fused stack unexpectedly contains parameters")
    if any(
        isinstance(module, TransformerBlock)
        for module in stack.modules()
    ):
        raise ValueError("fused stack contains a transformer block")

    expected_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "split_manifest_sha256": manifest_hash,
        "teacher_state_sha256": teacher_state_hash,
        "source_artifacts": expected_sources,
        "layer_provenance": {
            "layer_0": provenance_0,
            "layer_1": provenance_1,
        },
        "build_contract": {
            "operation": "algebraic_fusion_only",
            "weights_fitted_or_updated": False,
            "cross_layer_bypass_required": True,
            "test_used_for_build_or_selection": False,
        },
    }
    _assert_numeric_tree_match(
        metadata,
        expected_metadata,
        path="fused stack metadata",
    )
    artifact_hash_before = _sha256(paths.stack)
    expected_fused_artifact = {
        "filename": paths.stack.name,
        "sha256": artifact_hash_before,
        "config": asdict(config),
        "metadata": metadata,
    }
    _assert_numeric_tree_match(
        report.get("fused_artifact"),
        expected_fused_artifact,
        path="fused executor fused_artifact",
    )

    raw_lazy_artifact = torch.load(
        paths.runtime,
        map_location="cpu",
        weights_only=True,
    )
    raw_lazy = _object(
        raw_lazy_artifact,
        "lazy fused runtime artifact",
    )
    expected_lazy_payload_fields = {
        "format_version",
        "artifact_kind",
        "config",
        "state_dict",
        "sidecars",
        "metadata",
    }
    if set(raw_lazy) != expected_lazy_payload_fields:
        raise ValueError("lazy fused runtime artifact fields mismatch")
    if _integer(
        raw_lazy.get("format_version"),
        "lazy fused runtime format_version",
    ) != 2:
        raise ValueError("unsupported lazy fused runtime artifact format")
    if (
        raw_lazy.get("artifact_kind")
        != "lazy_fused_two_layer_modal_stack"
    ):
        raise ValueError("lazy fused runtime artifact kind mismatch")
    expected_lazy_state_names = {
        "first_input_mean",
        "first_input_kernel",
        "first_hidden_bias",
        "bridge_kernel",
        "bridge_bias",
        "second_fused_output_weight",
        "second_fused_output_bias",
    }
    raw_lazy_state = _object(
        raw_lazy.get("state_dict"),
        "lazy fused runtime state_dict",
    )
    if set(raw_lazy_state) != expected_lazy_state_names:
        raise ValueError("lazy fused runtime state fields mismatch")
    fast_state_tensor_bytes = 0
    for name, value in raw_lazy_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"lazy fused runtime state is not a tensor: {name}"
            )
        if not value.is_floating_point():
            raise ValueError(
                f"lazy fused runtime state is not floating point: {name}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"lazy fused runtime state is nonfinite: {name}"
            )
        fast_state_tensor_bytes += (
            value.numel() * value.element_size()
        )
    if fast_state_tensor_bytes != 199_808:
        raise ValueError("lazy fused runtime fast-state bytes mismatch")
    raw_sidecars = _object(
        raw_lazy.get("sidecars"),
        "lazy fused runtime sidecars",
    )
    _assert_numeric_tree_match(
        raw_sidecars,
        expected_sidecar_descriptors,
        path="lazy fused runtime sidecars",
    )

    unavailable_sidecar_root = (
        directory / ".verify-fast-runtime-has-no-sidecars"
    )
    if unavailable_sidecar_root.exists():
        raise ValueError(
            "reserved verifier sidecar-absence path unexpectedly exists"
        )
    lazy_stack, lazy_config, lazy_metadata = (
        load_lazy_fused_modal_stack(
            paths.runtime,
            sidecar_root=unavailable_sidecar_root,
        )
    )
    if lazy_config != config:
        raise ValueError(
            "lazy fused runtime config differs from monolithic config"
        )
    if lazy_metadata != metadata:
        raise ValueError(
            "lazy fused runtime metadata differs from monolithic metadata"
        )
    if _json_normalized(raw_lazy.get("config")) != _json_normalized(
        asdict(lazy_config)
    ):
        raise ValueError("lazy fused runtime raw config mismatch")
    if _json_normalized(raw_lazy.get("metadata")) != _json_normalized(
        lazy_metadata
    ):
        raise ValueError("lazy fused runtime raw metadata mismatch")
    if set(lazy_stack.state_dict()) != expected_lazy_state_names:
        raise ValueError("lazy fused runtime loaded state fields mismatch")
    for name, value in lazy_stack.state_dict().items():
        if not torch.equal(value.cpu(), raw_lazy_state[name]):
            raise ValueError(
                f"lazy fused loader changed fast tensor: {name}"
            )
    for name in (
        "checkpoint_sha256",
        "fisher_sha256",
        "teacher_state_sha256",
    ):
        value = lazy_metadata.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"lazy fused runtime provenance hash is invalid: {name}"
            )
    expected_fast_tensors = {
        "first_input_mean": stack.first.input_mean,
        "first_input_kernel": stack.first.input_kernel,
        "first_hidden_bias": stack.first.hidden_bias,
        "bridge_kernel": stack.bridge_kernel,
        "bridge_bias": stack.bridge_bias,
        "second_fused_output_weight": (
            stack.second.fused_output_weight
        ),
        "second_fused_output_bias": stack.second.fused_output_bias,
    }
    for name, expected in expected_fast_tensors.items():
        if not torch.equal(getattr(lazy_stack, name), expected):
            raise ValueError(
                f"lazy fused fast tensor differs from monolithic: {name}"
            )
    future = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    for name in ("first_input_kernel", "bridge_kernel"):
        if torch.count_nonzero(getattr(lazy_stack, name)[future]).item():
            raise ValueError(
                f"lazy fused runtime has noncausal weights: {name}"
            )
    if lazy_stack.fast_state_bytes != 199_808:
        raise ValueError("lazy fused runtime resident fast bytes mismatch")
    initial_lazy_status = _lazy_status_dict(lazy_stack)
    _validate_lazy_status(
        initial_lazy_status,
        path="loaded lazy fused runtime initial status",
        require_unloaded_fast_only=True,
    )
    if list(lazy_stack.parameters()):
        raise ValueError("lazy fused runtime unexpectedly contains parameters")
    if any(
        isinstance(module, TransformerBlock)
        for module in lazy_stack.modules()
    ):
        raise ValueError("lazy fused runtime contains a transformer block")

    lazy_artifact_hash_before = _sha256(paths.runtime)
    expected_lazy_artifact = {
        "filename": paths.runtime.name,
        "sha256": lazy_artifact_hash_before,
        "config": asdict(lazy_config),
        "metadata": lazy_metadata,
        "format_version": 2,
        "artifact_kind": "lazy_fused_two_layer_modal_stack",
        "fast_state_tensor_bytes": 199_808,
        "sidecar_descriptors": expected_sidecar_descriptors,
    }
    _assert_numeric_tree_match(
        report.get("lazy_fused_artifact"),
        expected_lazy_artifact,
        path="fused executor lazy_fused_artifact",
    )

    gate = {
        "maximum_absolute_nll_delta": 1e-6,
        "maximum_mean_answer_kl": 1e-6,
        "maximum_answer_logit_difference": 5e-4,
    }
    expected_protocol = {
        "operation": "algebraic_fusion_of_locked_modal_artifacts",
        "validation_split": "validation_fisher",
        "evaluation_split": "test",
        "test_used_for_build_or_selection": False,
        "fused_artifact_saved_and_reloaded_before_validation": True,
        "fused_artifact_saved_and_reloaded_before_test": True,
        "lazy_artifact_saved_and_reloaded_before_validation": True,
        "lazy_artifact_saved_and_reloaded_before_test": True,
        "validation_test_and_benchmark_runtime": (
            "fresh_lazy_fast_runtime"
        ),
        "instrumentation_runtime": "separate_fresh_lazy_runtime",
        "fast_runtime_sidecar_loads_during_validation_test_benchmark": 0,
        "validation_gate": gate,
        "validation_gate_passed": True,
        "test_evaluated_once_after_gate": True,
    }
    _assert_numeric_tree_match(
        report.get("protocol"),
        expected_protocol,
        path="fused executor protocol",
    )

    teacher = ToyTransformer(model_config)
    teacher.load_state_dict(model.state_dict())
    _fused_freeze(teacher)
    unfused = ToyTransformer(model_config)
    unfused.load_state_dict(model.state_dict())
    unfused.replace_layer(0, completed_0)
    unfused.replace_layer(1, completed_1)
    _fused_freeze(unfused)
    monolithic = FusedToyTransformer.from_teacher(teacher, stack)
    _fused_freeze(monolithic)
    if list(monolithic.parameters()):
        raise ValueError(
            "fused full runtime unexpectedly contains parameters"
        )
    if any(
        isinstance(module, TransformerBlock)
        for module in monolithic.modules()
    ):
        raise ValueError("fused full runtime contains a transformer block")

    lazy = FusedToyTransformer.from_teacher(teacher, lazy_stack)
    _fused_freeze(lazy)
    if list(lazy.parameters()):
        raise ValueError(
            "lazy fused full runtime unexpectedly contains parameters"
        )
    if any(
        isinstance(module, TransformerBlock)
        for module in lazy.modules()
    ):
        raise ValueError(
            "lazy fused full runtime contains a transformer block"
        )
    lazy_state_hash_before = _module_state_sha256(lazy_stack)
    status_before_validation = _lazy_status_dict(lazy_stack)
    _validate_lazy_status(
        status_before_validation,
        path="lazy fast status before validation",
        require_unloaded_fast_only=True,
    )

    trace_stack, trace_config, trace_metadata = (
        load_lazy_fused_modal_stack(paths.runtime)
    )
    if trace_config != lazy_config or trace_metadata != lazy_metadata:
        raise ValueError("fresh lazy trace runtime differs from fast runtime")
    trace_runtime = FusedToyTransformer.from_teacher(
        teacher,
        trace_stack,
    )
    _fused_freeze(trace_runtime)
    trace_contract = _fused_trace_contract(
        validation_inputs=splits.validation.input_ids[:8],
        unfused=unfused,
        lazy=trace_runtime,
        lazy_stack=trace_stack,
        expected_sidecar_file_bytes=expected_sidecar_file_bytes,
    )
    _assert_numeric_tree_match(
        report.get("dispatch_and_trace_contract"),
        trace_contract,
        path="fused executor dispatch_and_trace_contract",
    )
    validation = _fused_evaluate_split(
        split=splits.validation,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=lazy,
    )
    if not (
        _fused_gate_passed(
            validation["monolithic_vs_unfused"],
            gate,
        )
        and _fused_gate_passed(
            validation["lazy_vs_unfused"],
            gate,
        )
        and _object(
            validation["lazy_vs_monolithic"],
            "lazy validation monolithic comparison",
        ).get("logits_bit_exact")
    ):
        raise ValueError(
            "fused executor recomputed validation gate failed"
        )
    _assert_numeric_tree_match(
        report.get("validation"),
        validation,
        path="fused executor validation",
    )
    status_after_validation = _lazy_status_dict(lazy_stack)
    test = _fused_evaluate_split(
        split=splits.test,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=lazy,
    )
    _assert_numeric_tree_match(
        report.get("test"),
        test,
        path="fused executor test",
    )
    status_after_test = _lazy_status_dict(lazy_stack)
    reported_fast_status = _object(
        report.get("lazy_fast_runtime_status"),
        "fused executor lazy_fast_runtime_status",
    )
    if set(reported_fast_status) != {
        "before_validation",
        "after_validation",
        "after_test",
        "after_benchmark",
        "zero_sidecar_loads_throughout",
    }:
        raise ValueError("lazy fast runtime status fields mismatch")
    for label, actual in (
        ("before_validation", status_before_validation),
        ("after_validation", status_after_validation),
        ("after_test", status_after_test),
    ):
        _validate_lazy_status(
            reported_fast_status.get(label),
            path=f"reported lazy status {label}",
            require_unloaded_fast_only=True,
        )
        _assert_numeric_tree_match(
            reported_fast_status.get(label),
            actual,
            path=f"fused executor lazy status {label}",
        )
    status_after_benchmark = _validate_lazy_status(
        reported_fast_status.get("after_benchmark"),
        path="reported lazy status after_benchmark",
        require_unloaded_fast_only=True,
    )
    if (
        status_after_benchmark["fast_path_calls"]
        <= status_after_test["fast_path_calls"]
        or status_after_benchmark["last_dispatch"]
        != "fast_cross_layer"
        or reported_fast_status.get("zero_sidecar_loads_throughout")
        is not True
    ):
        raise ValueError(
            "reported lazy benchmark status does not prove fast-only dispatch"
        )

    arithmetic = _fused_expected_arithmetic(
        stack=stack,
        model_config=model_config,
        completed_layers=(completed_0, completed_1),
    )
    if report_format_version == 2:
        arithmetic["triangular_interpretation"] = (
            "nonzero causal arithmetic available to a sparse or "
            "triangular backend"
        )
    _assert_numeric_tree_match(
        report.get("arithmetic"),
        arithmetic,
        path="fused executor arithmetic",
    )
    lazy_storage_contract = {
        "fast_stack_resident_tensor_bytes": 199_808,
        "sidecar_resident_tensor_bytes": 203_648,
        "model_shell_tensor_bytes": 6_144,
        "default_full_runtime_resident_tensor_bytes": 205_952,
        "loaded_full_runtime_resident_tensor_bytes": 409_600,
    }
    if (
        _fused_state_storage(lazy_stack)["total_state_bytes"]
        != 199_808
        or _fused_state_storage(lazy)["total_state_bytes"]
        != 205_952
        or _fused_state_storage(
            torch.nn.ModuleList([completed_0, completed_1])
        )["total_state_bytes"]
        != 203_648
    ):
        raise ValueError("lazy fused runtime deterministic storage mismatch")
    storage = {
        "teacher_full_model": _fused_state_storage(teacher),
        "unfused_full_model": _fused_state_storage(unfused),
        "unfused_two_layer_executors": _fused_state_storage(
            torch.nn.ModuleList([completed_0, completed_1])
        ),
        "fused_full_model": _fused_state_storage(monolithic),
        "fused_two_layer_stack": _fused_state_storage(stack),
        "lazy_default_full_model_state": _fused_state_storage(lazy),
        "lazy_default_fast_stack_state": _fused_state_storage(
            lazy_stack
        ),
        "lazy_storage_contract": lazy_storage_contract,
        "fused_artifact_file_bytes": paths.stack.stat().st_size,
        "lazy_artifact_file_bytes": paths.runtime.stat().st_size,
        "instrumentation_sidecar_file_bytes": {
            name: path.stat().st_size
            for name, path in sidecar_source_paths.items()
        },
        "instrumentation_sidecar_total_file_bytes": (
            expected_sidecar_file_bytes
        ),
        "source_artifact_file_bytes": {
            name: path.stat().st_size
            for name, path in source_paths.items()
        },
    }
    _assert_numeric_tree_match(
        report.get("storage"),
        storage,
        path="fused executor storage",
    )
    benchmark_batch_count, minimum_speedup = _verify_fused_benchmark(
        report.get("benchmark_environment"),
        report.get("benchmark"),
    )
    lazy_benchmark_comparison = _fused_lazy_benchmark_comparison(
        report.get("benchmark")
    )
    _assert_numeric_tree_match(
        report.get("lazy_vs_monolithic_benchmark"),
        lazy_benchmark_comparison,
        path="fused executor lazy_vs_monolithic_benchmark",
    )
    triangular_summary: dict[str, object] = {}
    if report_format_version == 3:
        triangular_summary = _verify_fused_triangular_runtime(
            report.get("triangular_runtime_benchmark"),
            source_lazy_path=paths.runtime,
            source_lazy_sha256=lazy_artifact_hash_before,
            lazy_stack=lazy_stack,
            lazy_model=lazy,
            teacher=teacher,
            validation_split=splits.validation,
            gate=gate,
            arithmetic=arithmetic,
        )

    if _sha256(paths.stack) != artifact_hash_before:
        raise ValueError(
            "fused artifact changed during independent verification"
        )
    if _sha256(paths.runtime) != lazy_artifact_hash_before:
        raise ValueError(
            "lazy fused artifact changed during independent verification"
        )
    if {
        name: _sha256(path) for name, path in source_paths.items()
    } != source_hashes_before:
        raise ValueError(
            "fused source artifacts changed during verification"
        )
    if _module_state_sha256(model) != teacher_state_hash:
        raise ValueError(
            "fused executor verification mutated the teacher"
        )
    if _module_state_sha256(lazy_stack) != lazy_state_hash_before:
        raise ValueError(
            "lazy fused verification mutated the resident fast state"
        )
    fused_test_metrics = _object(
        _object(
            test.get("systems"),
            "fused test systems",
        ).get("lazy"),
        "fused test metrics",
    )
    fused_test_comparison = _object(
        test.get("lazy_vs_unfused"),
        "fused test comparison",
    )
    summary = {
        "present": True,
        "format_version": report_format_version,
        "cross_layer_bypass": True,
        "parameter_count": 0,
        "contains_transformer_block": False,
        "lazy_runtime": True,
        "fast_state_tensor_count": 7,
        "fast_stack_resident_tensor_bytes": 199_808,
        "sidecar_resident_tensor_bytes": 203_648,
        "default_full_runtime_resident_tensor_bytes": 205_952,
        "loaded_full_runtime_resident_tensor_bytes": 409_600,
        "zero_sidecar_loads_during_fast_evaluation": True,
        "sidecar_loaded_exactly_once": True,
        "instrumentation_cache_reused": True,
        "instrumentation_evicted": True,
        "validation_gate_passed": True,
        "dense_executed_multiplies": arithmetic[
            "fused_dense_executed_multiplies"
        ],
        "triangular_nonzero_multiplies": arithmetic[
            "fused_triangular_nonzero_multiplies"
        ],
        "dense_multiply_ratio": arithmetic[
            "fused_dense_vs_original_ratio"
        ],
        "benchmark_batch_count": benchmark_batch_count,
        "minimum_observed_fused_vs_unfused_speedup": minimum_speedup,
        "geometric_mean_lazy_to_monolithic_latency_ratio": (
            lazy_benchmark_comparison[
                "geometric_mean_lazy_to_monolithic_latency_ratio"
            ]
        ),
        "maximum_lazy_to_monolithic_latency_ratio": (
            lazy_benchmark_comparison[
                "maximum_lazy_to_monolithic_latency_ratio"
            ]
        ),
        "benchmark_hard_latency_gate_applied": False,
        "test_accuracy": _number(
            fused_test_metrics.get("answer_accuracy"),
            "fused test answer_accuracy",
        ),
        "test_paired_accuracy": _number(
            fused_test_metrics.get("paired_context_accuracy"),
            "fused test paired_context_accuracy",
        ),
        "test_nll": _number(
            fused_test_metrics.get("hard_nll"),
            "fused test hard_nll",
        ),
        "test_maximum_answer_logit_difference": _number(
            fused_test_comparison.get(
                "maximum_answer_logit_difference"
            ),
            "fused test maximum answer-logit difference",
        ),
        "status": "verified",
    }
    summary.update(triangular_summary)
    return summary


def _verify_optional_runtime_manifest(
    directory: Path,
    *,
    model: ToyTransformer,
    checkpoint_hash: str,
    sequence_length: int,
) -> dict[str, object]:
    path = directory / "runtime_manifest.json"
    if not path.is_file():
        if (directory / "fused_modal_runtime.pt").is_file():
            raise FileNotFoundError(
                "fused runtime requires runtime_manifest.json"
            )
        return {"present": False}

    manifest = load_runtime_manifest(path)
    if runtime_manifest_bytes(manifest) != path.read_bytes():
        raise ValueError("runtime manifest is not canonical JSON")
    regenerated = manifest_from_legacy_runtime(directory)
    if manifest != regenerated:
        raise ValueError(
            "runtime manifest does not match the authenticated legacy runtime"
        )
    if manifest.model.source_state_sha256 != _module_state_sha256(model):
        raise ValueError("runtime manifest source-model fingerprint mismatch")
    config_sha256 = hashlib.sha256(
        json.dumps(
            asdict(model.config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if manifest.model.source_config_sha256 != config_sha256:
        raise ValueError("runtime manifest source-config fingerprint mismatch")
    if manifest.resource("checkpoint").sha256 != checkpoint_hash:
        raise ValueError("runtime manifest checkpoint hash mismatch")
    expected_layers = tuple(
        f"layer.{index}" for index in range(len(model.layers))
    )
    if manifest.model.layer_ids != expected_layers:
        raise ValueError("runtime manifest source-layer catalog mismatch")
    if (
        manifest.sequence.policy != "fixed"
        or manifest.sequence.minimum_length != sequence_length
        or manifest.sequence.maximum_length != sequence_length
    ):
        raise ValueError("runtime manifest sequence guard mismatch")
    if any(
        segment.validation.status != "passed"
        for segment in manifest.segments
    ):
        raise ValueError("runtime manifest contains an unverified segment")

    total_resource_bytes = 0
    for descriptor in manifest.resources:
        with open_verified_resource(directory, descriptor):
            total_resource_bytes += descriptor.size_bytes
    return {
        "present": True,
        "schema_version": manifest.schema_version,
        "segment_count": len(manifest.segments),
        "resource_count": len(manifest.resources),
        "authenticated_resource_bytes": total_resource_bytes,
        "sequence_policy": manifest.sequence.policy,
        "sequence_length": manifest.sequence.minimum_length,
        "fallback_segment_count": sum(
            segment.fallback_policy == "source_model"
            for segment in manifest.segments
        ),
        "status": "verified",
    }


def verify_build(directory: Path) -> dict[str, object]:
    """Raise on any failed invariant and return a compact verification summary."""

    checkpoint_path = directory / "checkpoint.pt"
    fisher_path = directory / "fisher_modes.pt"
    report_path = directory / "fisher_report.json"
    manifest_path = directory / "split_manifest.json"
    metrics_path = directory / "training_metrics.jsonl"
    for path in (
        checkpoint_path,
        fisher_path,
        report_path,
        manifest_path,
        metrics_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    splits = build_associative_recall_splits(task_config)
    stored_split_ids = checkpoint["split_context_ids"]
    regenerated = {
        "train": splits.train.context_ids,
        "validation_fisher": splits.validation.context_ids,
        "test": splits.test.context_ids,
    }
    for name, context_ids in regenerated.items():
        if not torch.equal(stored_split_ids[name], context_ids):
            raise ValueError(f"checkpoint split mismatch: {name}")

    model = ToyTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate_associative_recall(
        model, splits.validation
    )
    test_metrics = evaluate_associative_recall(model, splits.test)
    if validation_metrics.answer_accuracy < 0.995:
        raise ValueError("reloaded checkpoint misses validation accuracy gate")
    if test_metrics.answer_accuracy < 0.995:
        raise ValueError("reloaded checkpoint misses test accuracy gate")
    if validation_metrics.paired_context_accuracy < 0.99:
        raise ValueError("reloaded checkpoint misses paired-context gate")

    bases, transitions, jacobians, metadata = load_fisher_build(fisher_path)
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Fisher artifact checkpoint hash mismatch")
    if len(bases) != 6 or len(transitions) != 2 or len(jacobians) != 2:
        raise ValueError("Fisher artifact has an unexpected component count")

    basis_summary: dict[str, dict[str, float | int]] = {}
    for name, basis in bases.items():
        if not torch.isfinite(basis.matrix).all():
            raise ValueError(f"nonfinite Fisher matrix: {name}")
        if not torch.isfinite(basis.eigenvalues).all():
            raise ValueError(f"nonfinite Fisher spectrum: {name}")
        if (basis.eigenvalues < 0).any():
            raise ValueError(f"negative Fisher eigenvalue: {name}")
        if (basis.eigenvalues[:-1] < basis.eigenvalues[1:]).any():
            raise ValueError(f"Fisher spectrum is not descending: {name}")
        identity = torch.eye(basis.width, dtype=basis.vectors.dtype)
        torch.testing.assert_close(
            basis.vectors.transpose(0, 1) @ basis.vectors,
            identity,
            rtol=1e-10,
            atol=1e-10,
        )
        reconstructed = (
            basis.vectors
            @ torch.diag(basis.eigenvalues)
            @ basis.vectors.transpose(0, 1)
        )
        torch.testing.assert_close(
            reconstructed,
            basis.matrix,
            rtol=1e-10,
            atol=1e-12,
        )
        if basis.fisher_trace <= 0:
            raise ValueError(f"collapsed Fisher trace: {name}")
        probe = torch.linspace(
            -1,
            1,
            basis.width,
            dtype=basis.mean.dtype,
        )
        torch.testing.assert_close(
            basis.reconstruct(basis.project(probe)),
            probe,
            rtol=1e-10,
            atol=1e-10,
        )
        basis_summary[name] = {
            "trace": basis.fisher_trace,
            "k90": basis.modes_for_fraction(0.90),
            "k99": basis.modes_for_fraction(0.99),
        }

    for transition in transitions:
        if not torch.isfinite(transition.weights).all():
            raise ValueError("nonfinite modal transition")
        if not torch.isfinite(transition.bias).all():
            raise ValueError("nonfinite modal transition bias")
    for jacobian in jacobians:
        if not torch.isfinite(jacobian.mean).all():
            raise ValueError("nonfinite modal Jacobian mean")
        if not torch.isfinite(jacobian.rms).all():
            raise ValueError("nonfinite modal Jacobian RMS")
        if (jacobian.rms < 0).any():
            raise ValueError("negative modal Jacobian RMS")
        if jacobian.samples <= 0:
            raise ValueError("empty modal Jacobian")

    report = json.loads(report_path.read_text())
    if report.get("model") != checkpoint["model_config"]:
        raise ValueError("JSON report model config mismatch")
    if report["artifacts"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("JSON report checkpoint hash mismatch")
    if not all(
        item["validation_passed"]
        for item in report["activation_modes"].values()
    ):
        raise ValueError("JSON report contains a failed decomposition")
    manifest = _object(
        json.loads(manifest_path.read_text()),
        "split manifest",
    )
    if manifest["train"]["examples"] != splits.train.samples:
        raise ValueError("split manifest train size mismatch")
    if manifest["validation_fisher"]["examples"] != splits.validation.samples:
        raise ValueError("split manifest validation size mismatch")
    if manifest["test"]["examples"] != splits.test.samples:
        raise ValueError("split manifest test size mismatch")
    if not metrics_path.read_text().strip():
        raise ValueError("training metrics log is empty")

    intervention_summary = _verify_optional_interventions(
        directory,
        checkpoint_hash=checkpoint_hash,
        fisher_path=fisher_path,
        bases=bases,  # type: ignore[arg-type]
        test_metrics=test_metrics,
        samples=splits.test.samples,
        contexts=splits.test.contexts,
        sequence_length=task_config.sequence_length,
    )
    modal_executor_summaries = {
        str(layer_index): _verify_optional_modal_executor(
            directory,
            artifact_layer_index=layer_index,
            checkpoint_hash=checkpoint_hash,
            fisher_path=fisher_path,
            bases=bases,
            model=model,
            model_config=model.config,
            splits=splits,
            manifest=manifest,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            sequence_length=task_config.sequence_length,
        )
        for layer_index in range(len(model.layers))
    }
    modal_completion_summaries = {
        str(layer_index): _verify_optional_modal_completion(
            directory,
            artifact_layer_index=layer_index,
            checkpoint_hash=checkpoint_hash,
            fisher_path=fisher_path,
            bases=bases,
            model=model,
            model_config=model.config,
            splits=splits,
            manifest=manifest,
            sequence_length=task_config.sequence_length,
        )
        for layer_index in range(len(model.layers))
    }
    modal_composition_summary = _verify_optional_modal_composition(
        directory,
        checkpoint_hash=checkpoint_hash,
        fisher_path=fisher_path,
        bases=bases,
        model=model,
        model_config=model.config,
        splits=splits,
        sequence_length=task_config.sequence_length,
    )
    fused_executor_summary = _verify_optional_fused_executor(
        directory,
        checkpoint_hash=checkpoint_hash,
        fisher_path=fisher_path,
        bases=bases,
        model=model,
        model_config=model.config,
        splits=splits,
        sequence_length=task_config.sequence_length,
    )
    runtime_manifest_summary = _verify_optional_runtime_manifest(
        directory,
        model=model,
        checkpoint_hash=checkpoint_hash,
        sequence_length=task_config.sequence_length,
    )
    modal_executor_summary = modal_executor_summaries["0"]
    modal_completion_summary = modal_completion_summaries["0"]
    return {
        "checkpoint_sha256": checkpoint_hash,
        "validation_accuracy": validation_metrics.answer_accuracy,
        "validation_paired_accuracy": (
            validation_metrics.paired_context_accuracy
        ),
        "test_accuracy": test_metrics.answer_accuracy,
        "basis_count": len(bases),
        "transition_count": len(transitions),
        "jacobian_count": len(jacobians),
        "bases": basis_summary,
        "interventions": intervention_summary,
        "modal_executor": modal_executor_summary,
        "modal_completion": modal_completion_summary,
        "modal_executors": modal_executor_summaries,
        "modal_completions": modal_completion_summaries,
        "modal_composition": modal_composition_summary,
        "fused_executor": fused_executor_summary,
        "runtime_manifest": runtime_manifest_summary,
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a trained Fisher compute-mode build."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("artifacts/associative_recall"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_build(args.directory), indent=2))


if __name__ == "__main__":
    main()
