"""Fit and evaluate frozen-teacher modal tail completion bridges."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .associative import (
    AssociativeRecallMetrics,
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from .adapters import ToyTransformerAdapter, as_model_adapter
from .config import TransformerConfig
from .modal_artifacts import (
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
    save_position_modal_completion,
)
from .modal_executor import (
    PositionConditionedModalBottleneckExecutor,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
    load_position_modal_executor,
)
from .modal_executor_experiment import (
    _activation_fit_metrics,
    _collect_activations,
    _estimated_block_multiplies,
    _estimated_modal_multiplies,
)
from .modes import FisherModeBasis, load_fisher_build
from .model import ToyTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _metrics(metrics: AssociativeRecallMetrics) -> dict[str, object]:
    return asdict(metrics)


@torch.no_grad()
def _replacement_logits(
    model: ToyTransformer | ToyTransformerAdapter,
    *,
    layer_index: int,
    executor,
    split,
) -> Tensor:
    adapter = as_model_adapter(model)
    if not isinstance(adapter.module, ToyTransformer):
        raise TypeError(
            "the associative-recall evaluator requires a ToyTransformer"
        )
    try:
        segment = adapter.segments[layer_index]
    except IndexError:
        raise ValueError("layer_index is outside the adapter segments") from None
    with adapter.replaced_segments({segment.id: executor}):
        return associative_recall_answer_logits(adapter.module, split)


def _behavior(
    split,
    logits: Tensor,
    teacher_logits: Tensor,
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
        "metrics": _metrics(metrics),
        "teacher_to_system_answer_kl": kl.mean().item(),
        "maximum_answer_kl": kl.max().item(),
    }


def _r_squared(actual: Tensor, predicted: Tensor) -> float:
    residual = (actual - predicted).square().sum()
    centered = (
        actual - actual.mean(dim=0, keepdim=True)
    ).square().sum()
    return 1.0 - (residual / centered).item() if centered > 0 else 1.0


@torch.no_grad()
def _completion_metrics(
    completion: PositionConditionedModalCompletion,
    mean_control: PositionConditionedModalCompletion,
    activations: Tensor,
    basis: FisherModeBasis,
    *,
    train_tail_scale: Tensor,
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
                (actual - predicted).square().sum() / centered_sum
            ).item()
        )
    centered_activations = (
        activations.to(torch.float64)
        - activations.to(torch.float64).mean(dim=0, keepdim=True)
    )
    return {
        "samples": activations.shape[0],
        "tail_r_squared": _r_squared(actual_tail, predicted_tail),
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
            (completion_mse / zero_mse).item() if zero_mse > 0 else 0.0
        ),
        "tail_mse_ratio_vs_mean": (
            (completion_mse / mean_mse).item() if mean_mse > 0 else 0.0
        ),
        "full_activation_r_squared": _r_squared(
            activations.to(torch.float64),
            reconstructed,
        ),
        "full_activation_rmse": activation_residual.square()
        .mean()
        .sqrt()
        .item(),
        "relative_centered_residual_norm": (
            activation_residual.norm() / centered_activations.norm()
        ).item(),
    }


def _train_tail_scale(
    activations: Tensor,
    basis: FisherModeBasis,
    *,
    kept_modes: int,
    minimum_scale: float,
) -> Tensor:
    full = basis.project(
        activations.to(torch.float64),
        modes=basis.width,
        centering="position",
    )
    return full[..., kept_modes:].std(dim=0).clamp_min(minimum_scale)


def _projection(
    basis: FisherModeBasis,
    modes: int,
) -> PositionConditionedModalProjection:
    return PositionConditionedModalProjection.from_basis(
        basis,
        modes=modes,
    )


def _completion_parameter_count(
    *completions: PositionConditionedModalCompletion,
) -> int:
    return sum(
        completion.graph.learned_parameter_count
        for completion in completions
    )


def _state_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in module.state_dict().values()
    )


def _write_markdown(
    path: Path,
    report: dict[str, object],
    *,
    input_completion: PositionConditionedModalCompletion,
    output_completion: PositionConditionedModalCompletion,
) -> None:
    validation = report["validation_ablations"]
    test = report["test_ablations"]
    assert isinstance(validation, dict)
    assert isinstance(test, dict)
    labels = {
        "teacher": "Frozen teacher",
        "input_truncation": "Input truncation only",
        "input_completion": "Input completion only",
        "output_truncation": "Output truncation only",
        "output_completion": "Output completion only",
        "both_truncations": "Both truncations",
        "mean_completion": "Fit-set mean-tail control",
        "both_completions": "Both learned completions",
        "oracle_round_trip": "Full-basis oracle round trip",
        "graph_zero_tail": "Standalone modal graph",
        "graph_output_completion": (
            "Standalone modal graph + output completion"
        ),
    }

    def rows(section: dict[str, object]) -> list[str]:
        result: list[str] = []
        for name, label in labels.items():
            item = section[name]
            assert isinstance(item, dict)
            metrics = item["metrics"]
            assert isinstance(metrics, dict)
            result.append(
                f"| {label} | "
                f"{float(metrics['answer_accuracy']):.3%} | "
                f"{float(metrics['paired_context_accuracy']):.3%} | "
                f"{float(metrics['hard_nll']):.6f} | "
                f"{float(item['teacher_to_system_answer_kl']):.6g} |"
            )
        return result

    selected = report["selected_configuration"]
    accounting = report["accounting"]
    local = report["local_completion_metrics"]
    assert isinstance(selected, dict)
    assert isinstance(accounting, dict)
    assert isinstance(local, dict)
    input_validation = local["input"]["validation"]  # type: ignore[index]
    output_validation = local["output"]["validation"]  # type: ignore[index]
    assert isinstance(input_validation, dict)
    assert isinstance(output_validation, dict)
    lines = [
        "# Conditional Modal Completion",
        "",
        "The transformer checkpoint stayed frozen. Only deterministic ridge",
        "maps from retained to discarded Fisher coordinates were fitted.",
        "",
        "## Validation ablations",
        "",
        "| System | Accuracy | Paired | Hard NLL | Teacher KL |",
        "|---|---:|---:|---:|---:|",
        *rows(validation),
        "",
        "## Exploratory test ablations",
        "",
        "| System | Accuracy | Paired | Hard NLL | Teacher KL |",
        "|---|---:|---:|---:|---:|",
        *rows(test),
        "",
        "## Locked bridge",
        "",
        f"- Input map: {selected['input_graph_kind']} "
        f"{input_completion.kept_modes} -> {input_completion.tail_modes}",
        f"- Output map: {selected['output_graph_kind']} "
        f"{output_completion.kept_modes} -> "
        f"{output_completion.tail_modes}",
        f"- Ridge: {selected['ridge']}",
        f"- Learned completion parameters: "
        f"{accounting['completion_learned_parameters']}",
        f"- Incremental multiplies versus zero-tail bottleneck: "
        f"{accounting['completion_incremental_multiplies']}",
        f"- Completed standalone-graph multiply ratio: "
        f"{float(accounting['completed_graph_multiply_ratio']):.3%}",
        "",
        "On validation, input-tail completion reached "
        f"R-squared {float(input_validation['tail_r_squared']):.9f}; "
        "output-tail completion reached "
        f"{float(output_validation['tail_r_squared']):.6f}. "
        "Both learned bridges beat zero-tail and fit-set mean-tail controls.",
        "",
        "The selected pair restored the frozen-layer interface without",
        "changing any teacher weight. The output bridge also improved the",
        "already standalone modal executor, showing that the completion is",
        "useful beyond the diagnostic bottleneck.",
        "",
        "These results remain exploratory: the validation split supplied the",
        "saved Fisher basis and the test split had been inspected in earlier",
        "work. This is conditional prediction of redundant tail coordinates,",
        "not guaranteed recovery of arbitrary discarded information.",
        "",
    ]
    path.write_text("\n".join(lines))


def run_modal_completion_build(
    *,
    artifact_dir: Path,
    layer_index: int = 0,
    input_modes: int | None = None,
    output_modes: int | None = None,
    ridge: float = 1e-4,
) -> dict[str, object]:
    """Fit, validation-select, save, reload, and evaluate completion bridges."""

    if layer_index not in (0, 1):
        raise ValueError("layer_index must be 0 or 1")
    started = time.perf_counter()
    executor_paths = modal_executor_artifact_paths(
        artifact_dir,
        layer_index,
    )
    completion_paths = modal_completion_artifact_paths(
        artifact_dir,
        layer_index,
    )
    checkpoint_path = artifact_dir / "checkpoint.pt"
    fisher_path = artifact_dir / "fisher_modes.pt"
    modal_executor_path = executor_paths.executor
    manifest = json.loads(
        (artifact_dir / "split_manifest.json").read_text()
    )
    checkpoint_hash = _sha256(checkpoint_path)
    fisher_hash = _sha256(fisher_path)
    modal_executor_hash = _sha256(modal_executor_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = TransformerConfig(**checkpoint["model_config"])
    model = ToyTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = ToyTransformerAdapter(model)
    teacher_state_before = _module_state_sha256(model)
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    splits = build_associative_recall_splits(task_config)
    bases, _, _, fisher_metadata = load_fisher_build(fisher_path)
    if fisher_metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Fisher artifact does not match the checkpoint")
    try:
        segment = adapter.segments[layer_index]
    except IndexError:
        raise ValueError("layer_index is outside the adapter segments") from None
    input_name = segment.input_site
    output_name = segment.output_site
    input_basis = bases[input_name]
    output_basis = bases[output_name]
    selected_input_modes = (
        input_basis.modes_for_fraction(0.99)
        if input_modes is None
        else input_modes
    )
    selected_output_modes = (
        output_basis.modes_for_fraction(0.99)
        if output_modes is None
        else output_modes
    )
    if not 1 <= selected_input_modes < input_basis.width:
        raise ValueError("input mode count cannot be completed")
    if not 1 <= selected_output_modes < output_basis.width:
        raise ValueError("output mode count cannot be completed")

    print("Collecting frozen-teacher boundary activations", flush=True)
    train_activations = _collect_activations(
        model,
        splits.train,
        (input_name, output_name),
    )
    validation_activations = _collect_activations(
        model,
        splits.validation,
        (input_name, output_name),
    )
    fit_config = ModalCompletionFitConfig(ridge=ridge)
    input_candidates: dict[
        str,
        tuple[PositionConditionedModalCompletion, object],
    ] = {}
    output_candidates: dict[
        str,
        tuple[PositionConditionedModalCompletion, object],
    ] = {}
    for name, shared in (("shared_local_linear", True), ("position_local_linear", False)):
        input_candidates[name] = fit_local_modal_completion(
            train_activations[input_name],
            input_basis,
            kept_modes=selected_input_modes,
            shared_weights=shared,
            fit_config=fit_config,
        )
        output_candidates[name] = fit_local_modal_completion(
            train_activations[output_name],
            output_basis,
            kept_modes=selected_output_modes,
            shared_weights=shared,
            fit_config=fit_config,
        )

    original_layer = adapter.source_module(segment.layer_ids[0])
    teacher_validation_logits = associative_recall_answer_logits(
        model,
        splits.validation,
    )
    teacher_validation_metrics = associative_recall_metrics_from_logits(
        splits.validation,
        teacher_validation_logits,
    )
    gate = {
        "minimum_answer_accuracy": 1.0,
        "minimum_paired_accuracy": 1.0,
        "maximum_nll_increase": 2e-5,
    }
    candidates: list[dict[str, object]] = []
    for input_kind, (input_completion, input_fit) in input_candidates.items():
        for output_kind, (
            output_completion,
            output_fit,
        ) in output_candidates.items():
            executor = PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                input_completion=input_completion,
                output_completion=output_completion,
            )
            logits = _replacement_logits(
                model,
                layer_index=layer_index,
                executor=executor,
                split=splits.validation,
            )
            behavior = _behavior(
                splits.validation,
                logits,
                teacher_validation_logits,
            )
            metrics = behavior["metrics"]
            assert isinstance(metrics, dict)
            passed = (
                float(metrics["answer_accuracy"])
                >= gate["minimum_answer_accuracy"]
                and float(metrics["paired_context_accuracy"])
                >= gate["minimum_paired_accuracy"]
                and float(metrics["hard_nll"])
                <= teacher_validation_metrics.hard_nll
                + gate["maximum_nll_increase"]
            )
            candidates.append(
                {
                    "input_graph_kind": input_kind,
                    "output_graph_kind": output_kind,
                    "input_fit": asdict(input_fit),  # type: ignore[arg-type]
                    "output_fit": asdict(output_fit),  # type: ignore[arg-type]
                    "completion_learned_parameters": (
                        _completion_parameter_count(
                            input_completion,
                            output_completion,
                        )
                    ),
                    "validation_behavior": behavior,
                    "validation_gate_passed": passed,
                }
            )
            print(
                f"  {input_kind} + {output_kind}: "
                f"accuracy={float(metrics['answer_accuracy']):.3%}, "
                f"paired={float(metrics['paired_context_accuracy']):.3%}, "
                f"NLL={float(metrics['hard_nll']):.9f}, "
                f"gate={'pass' if passed else 'fail'}",
                flush=True,
            )
    passing = [
        candidate
        for candidate in candidates
        if candidate["validation_gate_passed"]
    ]
    selection_pool = passing if passing else candidates
    selected = min(
        selection_pool,
        key=lambda candidate: (
            int(candidate["completion_learned_parameters"]),
            float(
                candidate["validation_behavior"]["metrics"]["hard_nll"]  # type: ignore[index]
            ),
        ),
    )
    input_kind = str(selected["input_graph_kind"])
    output_kind = str(selected["output_graph_kind"])
    input_completion = input_candidates[input_kind][0]
    output_completion = output_candidates[output_kind][0]
    mean_input = make_mean_modal_completion(
        train_activations[input_name],
        input_basis,
        kept_modes=selected_input_modes,
    )
    mean_output = make_mean_modal_completion(
        train_activations[output_name],
        output_basis,
        kept_modes=selected_output_modes,
    )
    full_input_projection = _projection(input_basis, input_basis.width)
    full_output_projection = _projection(output_basis, output_basis.width)
    kept_input_projection = _projection(
        input_basis,
        selected_input_modes,
    )
    kept_output_projection = _projection(
        output_basis,
        selected_output_modes,
    )
    saved_modal_executor, saved_config, saved_metadata = (
        load_position_modal_executor(modal_executor_path)
    )
    if (
        saved_config.input_modes != selected_input_modes
        or saved_config.output_modes != selected_output_modes
        or saved_metadata.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("saved modal executor is incompatible")

    systems = {
        "teacher": original_layer,
        "input_truncation": PositionConditionedModalBottleneckExecutor(
            original_layer,
            kept_input_projection,
            full_output_projection,
        ),
        "input_completion": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                input_completion=input_completion,
            )
        ),
        "output_truncation": PositionConditionedModalBottleneckExecutor(
            original_layer,
            full_input_projection,
            kept_output_projection,
        ),
        "output_completion": (
            PositionConditionedModalCompletionBottleneckExecutor(
                original_layer,
                output_completion=output_completion,
            )
        ),
        "both_truncations": PositionConditionedModalBottleneckExecutor(
            original_layer,
            kept_input_projection,
            kept_output_projection,
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
        "oracle_round_trip": PositionConditionedModalBottleneckExecutor(
            original_layer,
            full_input_projection,
            full_output_projection,
        ),
        "graph_zero_tail": saved_modal_executor,
        "graph_output_completion": (
            PositionConditionedCompletedModalGraphExecutor(
                saved_modal_executor,
                output_completion,
            )
        ),
    }

    validation_ablations: dict[str, object] = {}
    for name, executor in systems.items():
        logits = (
            teacher_validation_logits
            if name == "teacher"
            else _replacement_logits(
                model,
                layer_index=layer_index,
                executor=executor,
                split=splits.validation,
            )
        )
        validation_ablations[name] = _behavior(
            splits.validation,
            logits,
            teacher_validation_logits,
        )

    input_path = completion_paths.input_completion
    output_path = completion_paths.output_completion
    fit_protocol = {
        "fit_config": asdict(fit_config),
        "fit_split": "train",
        "selection_split": "validation_fisher",
        "coordinate_system": "full_position_centered_fisher_basis",
        "target": "discarded_tail_coordinates",
        "selection_rule": (
            "fewest_parameters_passing_behavior_gate_then_lowest_nll"
        ),
        "validation_gate": gate,
        "validation_is_fisher_informed": True,
        "test_used_for_fit_or_selection": False,
    }
    shared_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "modal_executor_sha256": modal_executor_hash,
        "layer_index": layer_index,
        "fit_context_ids_sha256": manifest["train"][
            "context_ids_sha256"
        ],
        "selection_context_ids_sha256": manifest[
            "validation_fisher"
        ]["context_ids_sha256"],
        "selected_candidate": selected,
        "fit_protocol": fit_protocol,
        "teacher_state_sha256": teacher_state_before,
    }
    save_position_modal_completion(
        input_path,
        completion=input_completion,
        metadata={
            **shared_metadata,
            "boundary_role": "input",
            "fit_activation": input_name,
        },
    )
    save_position_modal_completion(
        output_path,
        completion=output_completion,
        metadata={
            **shared_metadata,
            "boundary_role": "output",
            "fit_activation": output_name,
            "training_distribution": "clean_frozen_teacher_output",
        },
    )
    input_hash_before_test = _sha256(input_path)
    output_hash_before_test = _sha256(output_path)
    input_completion, input_config, input_metadata = (
        load_position_modal_completion(input_path)
    )
    output_completion, output_config, output_metadata = (
        load_position_modal_completion(output_path)
    )
    if (
        input_metadata["teacher_state_sha256"] != teacher_state_before
        or output_metadata["teacher_state_sha256"] != teacher_state_before
    ):
        raise ValueError("completion artifact teacher identity mismatch")

    # Rebuild deployment rows from the locked, reloaded artifacts before the
    # first test access in this run.
    systems["input_completion"] = (
        PositionConditionedModalCompletionBottleneckExecutor(
            original_layer,
            input_completion=input_completion,
        )
    )
    systems["output_completion"] = (
        PositionConditionedModalCompletionBottleneckExecutor(
            original_layer,
            output_completion=output_completion,
        )
    )
    systems["both_completions"] = (
        PositionConditionedModalCompletionBottleneckExecutor(
            original_layer,
            input_completion=input_completion,
            output_completion=output_completion,
        )
    )
    systems["graph_output_completion"] = (
        PositionConditionedCompletedModalGraphExecutor(
            saved_modal_executor,
            output_completion,
        )
    )
    print(
        f"Locked {input_config.graph_kind} input + "
        f"{output_config.graph_kind} output; evaluating test",
        flush=True,
    )
    teacher_test_logits = associative_recall_answer_logits(
        model,
        splits.test,
    )
    test_ablations: dict[str, object] = {}
    for name, executor in systems.items():
        logits = (
            teacher_test_logits
            if name == "teacher"
            else _replacement_logits(
                model,
                layer_index=layer_index,
                executor=executor,
                split=splits.test,
            )
        )
        test_ablations[name] = _behavior(
            splits.test,
            logits,
            teacher_test_logits,
        )

    test_activations = _collect_activations(
        model,
        splits.test,
        (input_name, output_name),
    )
    input_scale = _train_tail_scale(
        train_activations[input_name],
        input_basis,
        kept_modes=selected_input_modes,
        minimum_scale=fit_config.minimum_scale,
    )
    output_scale = _train_tail_scale(
        train_activations[output_name],
        output_basis,
        kept_modes=selected_output_modes,
        minimum_scale=fit_config.minimum_scale,
    )
    local_metrics = {
        "input": {
            "validation": _completion_metrics(
                input_completion,
                mean_input,
                validation_activations[input_name],
                input_basis,
                train_tail_scale=input_scale,
            ),
            "test": _completion_metrics(
                input_completion,
                mean_input,
                test_activations[input_name],
                input_basis,
                train_tail_scale=input_scale,
            ),
        },
        "output": {
            "validation": _completion_metrics(
                output_completion,
                mean_output,
                validation_activations[output_name],
                output_basis,
                train_tail_scale=output_scale,
            ),
            "test": _completion_metrics(
                output_completion,
                mean_output,
                test_activations[output_name],
                output_basis,
                train_tail_scale=output_scale,
            ),
        },
    }
    activation_fits = {
        name: _activation_fit_metrics(
            executor,
            validation_activations[input_name],
            validation_activations[output_name],
            prefix=f"layer.{layer_index}",
        )
        for name, executor in systems.items()
        if name != "teacher"
    }
    completion_parameters = _completion_parameter_count(
        input_completion,
        output_completion,
    )
    completion_map_multiplies = (
        input_completion.graph.edge_count
        + output_completion.graph.edge_count
    )
    tail_decode_multiplies = task_config.sequence_length * model_config.d_model * (
        input_completion.tail_modes + output_completion.tail_modes
    )
    completion_increment = (
        completion_map_multiplies + tail_decode_multiplies
    )
    original_multiplies = _estimated_block_multiplies(
        model_config,
        sequence_length=task_config.sequence_length,
    )
    graph_edges = saved_modal_executor.graph.edge_count
    base_graph_multiplies = _estimated_modal_multiplies(
        sequence_length=task_config.sequence_length,
        width=model_config.d_model,
        input_modes=selected_input_modes,
        output_modes=selected_output_modes,
        graph_edges=graph_edges,
    )
    graph_output_increment = (
        output_completion.graph.edge_count
        + task_config.sequence_length
        * model_config.d_model
        * output_completion.tail_modes
    )
    teacher_state_after = _module_state_sha256(model)
    if teacher_state_after != teacher_state_before:
        raise RuntimeError("modal completion mutated the frozen teacher")
    if (
        _sha256(input_path) != input_hash_before_test
        or _sha256(output_path) != output_hash_before_test
    ):
        raise RuntimeError("completion artifacts changed after test lock")

    report: dict[str, object] = {
        "format_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "modal_executor_sha256": modal_executor_hash,
        "input_completion_sha256": input_hash_before_test,
        "output_completion_sha256": output_hash_before_test,
        "teacher_state_sha256_before": teacher_state_before,
        "teacher_state_sha256_after": teacher_state_after,
        "teacher_was_frozen": True,
        "layer_index": layer_index,
        "input_activation": input_name,
        "output_activation": output_name,
        "input_modes": selected_input_modes,
        "output_modes": selected_output_modes,
        "fit_protocol": fit_protocol,
        "validation_candidates": candidates,
        "selected_configuration": {
            "input_graph_kind": input_config.graph_kind,
            "output_graph_kind": output_config.graph_kind,
            "ridge": ridge,
            "completion_learned_parameters": completion_parameters,
            "validation_gate_passed": selected[
                "validation_gate_passed"
            ],
        },
        "validation_ablations": validation_ablations,
        "test_ablations": test_ablations,
        "local_completion_metrics": local_metrics,
        "validation_layer_output_fits": activation_fits,
        "accounting": {
            "completion_learned_parameters": completion_parameters,
            "input_completion_parameters": (
                input_completion.graph.learned_parameter_count
            ),
            "output_completion_parameters": (
                output_completion.graph.learned_parameter_count
            ),
            "completion_state_bytes": (
                _state_bytes(input_completion)
                + _state_bytes(output_completion)
            ),
            "completion_map_multiplies": completion_map_multiplies,
            "additional_full_decode_multiplies": tail_decode_multiplies,
            "completion_incremental_multiplies": completion_increment,
            "original_block_estimated_multiplies": original_multiplies,
            "completed_bottleneck_estimated_multiplies": (
                original_multiplies
                + task_config.sequence_length
                * model_config.d_model
                * (
                    selected_input_modes
                    + input_basis.width
                    + selected_output_modes
                    + output_basis.width
                )
                + completion_map_multiplies
            ),
            "base_modal_graph_estimated_multiplies": base_graph_multiplies,
            "completed_modal_graph_estimated_multiplies": (
                base_graph_multiplies + graph_output_increment
            ),
            "completed_graph_multiply_ratio": (
                (base_graph_multiplies + graph_output_increment)
                / original_multiplies
            ),
        },
        "artifacts": {
            "input_completion": input_path.name,
            "output_completion": output_path.name,
            "modal_executor": modal_executor_path.name,
            "checkpoint": checkpoint_path.name,
            "fisher": fisher_path.name,
        },
        "artifact_hashes_locked_before_test": True,
        "scientific_status": (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = completion_paths.report_json
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _write_markdown(
        completion_paths.report_markdown,
        report,
        input_completion=input_completion,
        output_completion=output_completion,
    )
    selected_test = test_ablations["both_completions"]
    assert isinstance(selected_test, dict)
    selected_test_metrics = selected_test["metrics"]
    assert isinstance(selected_test_metrics, dict)
    print(
        "Modal completion complete: "
        f"test accuracy="
        f"{float(selected_test_metrics['answer_accuracy']):.3%}, "
        f"paired="
        f"{float(selected_test_metrics['paired_context_accuracy']):.3%}, "
        f"NLL={float(selected_test_metrics['hard_nll']):.9f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit causal modal tail completion around a frozen transformer "
            "layer."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/associative_recall"),
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--input-modes", type=int)
    parser.add_argument("--output-modes", type=int)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()
    run_modal_completion_build(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        input_modes=args.input_modes,
        output_modes=args.output_modes,
        ridge=args.ridge,
    )


if __name__ == "__main__":
    main()
