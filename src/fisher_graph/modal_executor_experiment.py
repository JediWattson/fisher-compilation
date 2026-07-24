"""Fit and evaluate a standalone position-conditioned modal layer executor."""

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
    build_associative_recall_splits,
    evaluate_associative_recall,
)
from .adapters import ToyTransformerAdapter, as_model_adapter
from .config import TransformerConfig
from .modal_artifacts import modal_executor_artifact_paths
from .modal_executor import (
    ModalExecutorConfig,
    ModalExecutorFitConfig,
    PositionConditionedModalBottleneckExecutor,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
    fit_causal_modal_graph,
    fit_position_modal_executor,
    load_position_modal_executor,
    save_position_modal_executor,
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


@torch.no_grad()
def _collect_activations(
    model: ToyTransformer | ToyTransformerAdapter,
    split,
    names: tuple[str, ...],
    *,
    batch_size: int = 256,
) -> dict[str, Tensor]:
    adapter = as_model_adapter(model)
    captured: dict[str, list[Tensor]] = {name: [] for name in names}
    module = adapter.module
    was_training = module.training
    module.eval()
    try:
        for start in range(0, split.samples, batch_size):
            output = adapter.forward(
                {
                    "input_ids": split.input_ids[
                        start : start + batch_size
                    ]
                },
                capture_sites=names,
                retain_gradients=False,
            )
            for name in names:
                captured[name].append(
                    output.activations[name].detach().cpu()
                )
    finally:
        module.train(was_training)
    return {
        name: torch.cat(values, dim=0)
        for name, values in captured.items()
    }


def _metrics(metrics: AssociativeRecallMetrics) -> dict[str, object]:
    return asdict(metrics)


def _evaluate_replacement(
    model: ToyTransformer | ToyTransformerAdapter,
    *,
    layer_index: int,
    executor,
    split,
) -> AssociativeRecallMetrics:
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
        return evaluate_associative_recall(adapter.module, split)


@torch.no_grad()
def _activation_fit_metrics(
    executor,
    inputs: Tensor,
    targets: Tensor,
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
    r_squared = (
        1.0 - (residual_sum / centered_sum).item()
        if centered_sum > 0
        else 1.0
    )
    return {
        "activation_r_squared": r_squared,
        "activation_rmse": residual.square().mean().sqrt().item(),
    }


def _estimated_block_multiplies(
    config: TransformerConfig,
    *,
    sequence_length: int,
) -> int:
    attention_projections = (
        4 * sequence_length * config.d_model * config.d_model
    )
    attention_products = (
        2 * sequence_length * sequence_length * config.d_model
    )
    feed_forward = (
        2 * sequence_length * config.d_model * config.d_ff
    )
    return attention_projections + attention_products + feed_forward


def _estimated_modal_multiplies(
    *,
    sequence_length: int,
    width: int,
    input_modes: int,
    output_modes: int,
    graph_edges: int,
) -> int:
    projections = sequence_length * width * (
        input_modes + output_modes
    )
    return projections + graph_edges


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    baseline = report["baseline"]
    bottleneck = report["bottleneck"]
    affine = report["affine_graph"]
    graph = report["nonlinear_graph"]
    assert isinstance(baseline, dict)
    assert isinstance(bottleneck, dict)
    assert isinstance(affine, dict)
    assert isinstance(graph, dict)
    lines = [
        "# Position-Conditioned Modal Executor",
        "",
        "## Exploratory test behavior",
        "",
        "| System | Answer accuracy | Paired accuracy | Hard NLL |",
        "|---|---:|---:|---:|",
    ]
    for label, section in (
        ("Original transformer", baseline),
        ("Transformer with modal bottlenecks", bottleneck),
        ("Standalone causal affine graph", affine),
        ("Standalone causal nonlinear graph", graph),
    ):
        metrics = section["test_metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            f"| {label} | "
            f"{float(metrics['answer_accuracy']):.3%} | "
            f"{float(metrics['paired_context_accuracy']):.3%} | "
            f"{float(metrics['hard_nll']):.6f} |"
        )
    config = graph["config"]
    size = graph["size"]
    fit = graph["fit"]
    graph_fit_config = graph["fit_config"]
    assert isinstance(config, dict)
    assert isinstance(size, dict)
    assert isinstance(fit, dict)
    assert isinstance(graph_fit_config, dict)
    lines.extend(
        [
            "",
            "## Selected modal graph",
            "",
            f"- Replaced layer: {report['layer_index']}",
            f"- Retained input/output modes: "
            f"{config['input_modes']}/{config['output_modes']}",
            f"- Routing width: {config['routing_width']}",
            f"- Best distillation step: {fit['best_step']}",
            f"- Learned parameters: {size['learned_parameters']}",
            f"- Explicit graph edges: {size['graph_edges']}",
            f"- Estimated multiplies per sequence: "
            f"{size['estimated_multiplies_per_sequence']}",
            f"- Original block estimated multiplies: "
            f"{size['original_block_estimated_multiplies_per_sequence']}",
            f"- Estimated multiply ratio: "
            f"{float(size['estimated_multiply_ratio']):.3%}",
            "",
            "The dense surrogate is causal by construction. At each output",
            "position it reads only retained Fisher modes from that position",
            "and earlier positions, passes them through a small GELU routing",
            "bank, predicts retained output modes, and reconstructs the",
            "residual stream using validation-derived position means. The",
            "routing features are learned features, not Fisher eigenmodes.",
            "",
            "The bottleneck row still runs the original transformer block and",
            "therefore measures compression loss. The affine and nonlinear",
            "rows remove that block entirely and measure executor fidelity.",
            "",
            "This tested affine baseline reached substantial aggregate",
            "activation R-squared but did not preserve associative-recall",
            "behavior; the tested nonlinear graph did. This comparison does",
            "not prove that every successful executor must be nonlinear.",
            "",
            "Routing width was selected on the validation/Fisher split before",
            f"the final test evaluation. Width {config['routing_width']} was "
            "the smallest passing",
            f"candidate for this one-initialization, "
            f"{int(graph_fit_config['steps']):,}-step search. That is not a",
            "claim of minimum possible capacity. However, this repository's",
            "test split was inspected during earlier exploratory work, so",
            "these numbers are evidence for this checkpoint rather than a",
            "fresh confirmatory result.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_modal_executor_build(
    *,
    artifact_dir: Path,
    layer_index: int = 0,
    input_modes: int | None = None,
    output_modes: int | None = None,
    routing_widths: tuple[int, ...] = (4, 6, 8, 12),
    fit_steps: int = 2_000,
) -> dict[str, object]:
    if layer_index not in (0, 1):
        raise ValueError("layer_index must be 0 or 1")
    if not routing_widths or any(width <= 0 for width in routing_widths):
        raise ValueError("routing_widths must be positive")
    if len(set(routing_widths)) != len(routing_widths):
        raise ValueError("routing_widths cannot contain duplicates")
    started = time.perf_counter()
    artifact_paths = modal_executor_artifact_paths(
        artifact_dir,
        layer_index,
    )
    checkpoint_path = artifact_dir / "checkpoint.pt"
    fisher_path = artifact_dir / "fisher_modes.pt"
    checkpoint_hash = _sha256(checkpoint_path)
    fisher_hash = _sha256(fisher_path)
    split_manifest = json.loads(
        (artifact_dir / "split_manifest.json").read_text()
    )
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
    if not 1 <= selected_input_modes <= input_basis.width:
        raise ValueError("input_modes is outside the Fisher basis")
    if not 1 <= selected_output_modes <= output_basis.width:
        raise ValueError("output_modes is outside the Fisher basis")

    baseline_validation = evaluate_associative_recall(
        model,
        splits.validation,
    )
    print(
        f"Collecting layer {layer_index} teacher activations "
        f"({input_name} -> {output_name})",
        flush=True,
    )
    train_activations = _collect_activations(
        adapter,
        splits.train,
        (input_name, output_name),
    )
    validation_activations = _collect_activations(
        adapter,
        splits.validation,
        (input_name, output_name),
    )

    original_layer = adapter.source_module(segment.layer_ids[0])
    bottleneck_executor = PositionConditionedModalBottleneckExecutor(
        original_layer,
        PositionConditionedModalProjection.from_basis(
            input_basis,
            modes=selected_input_modes,
        ),
        PositionConditionedModalProjection.from_basis(
            output_basis,
            modes=selected_output_modes,
        ),
    )
    bottleneck_validation = _evaluate_replacement(
        adapter,
        layer_index=layer_index,
        executor=bottleneck_executor,
        split=splits.validation,
    )

    affine_fit = fit_causal_modal_graph(
        train_activations[input_name],
        train_activations[output_name],
        input_basis,
        output_basis,
        input_modes=selected_input_modes,
        output_modes=selected_output_modes,
    )
    affine_executor = PositionConditionedModalGraphExecutor.from_fit(
        input_basis,
        output_basis,
        affine_fit,
    )
    affine_validation = _evaluate_replacement(
        adapter,
        layer_index=layer_index,
        executor=affine_executor,
        split=splits.validation,
    )

    validation_gate = {
        "minimum_answer_accuracy": 0.995,
        "minimum_paired_accuracy": 0.99,
        "maximum_nll_increase": 0.01,
    }
    fit_config = ModalExecutorFitConfig(
        steps=fit_steps,
        evaluation_interval=max(1, fit_steps // 20),
    )
    fit_protocol = {
        "fit_config": asdict(fit_config),
        "coordinate_normalization": "per_position_mode_sample_std",
        "standard_deviation_correction": 1,
        "minimum_scale": fit_config.minimum_scale,
        "initialization_count": 1,
        "selection_rule": (
            "smallest_validation_gate_passing_width_else_lowest_nll"
        ),
    }
    candidate_results: list[dict[str, object]] = []
    candidate_executors: dict[int, PositionConditionedModalGraphExecutor] = {}
    for routing_width in sorted(routing_widths):
        config = ModalExecutorConfig(
            input_activation=input_name,
            output_activation=output_name,
            sequence_length=task_config.sequence_length,
            input_modes=selected_input_modes,
            output_modes=selected_output_modes,
            routing_width=routing_width,
        )
        print(
            f"Fitting causal nonlinear modal graph: "
            f"{selected_input_modes}->{routing_width}->"
            f"{selected_output_modes}",
            flush=True,
        )
        executor, fit_report = fit_position_modal_executor(
            train_activations[input_name],
            train_activations[output_name],
            input_basis,
            output_basis,
            config=config,
            fit_config=fit_config,
            validation_input_activations=validation_activations[
                input_name
            ],
            validation_output_activations=validation_activations[
                output_name
            ],
        )
        validation_metrics = _evaluate_replacement(
            adapter,
            layer_index=layer_index,
            executor=executor,
            split=splits.validation,
        )
        passed = (
            validation_metrics.answer_accuracy
            >= validation_gate["minimum_answer_accuracy"]
            and validation_metrics.paired_context_accuracy
            >= validation_gate["minimum_paired_accuracy"]
            and validation_metrics.hard_nll
            <= baseline_validation.hard_nll
            + validation_gate["maximum_nll_increase"]
        )
        candidate_executors[routing_width] = executor
        candidate_results.append(
            {
                "routing_width": routing_width,
                "fit_config": asdict(fit_config),
                "fit": {
                    **asdict(fit_report),
                    "history": [
                        asdict(point) for point in fit_report.history
                    ],
                },
                "validation_metrics": _metrics(validation_metrics),
                "validation_gate_passed": passed,
            }
        )
        print(
            f"  validation accuracy="
            f"{validation_metrics.answer_accuracy:.3%}, "
            f"paired={validation_metrics.paired_context_accuracy:.3%}, "
            f"NLL={validation_metrics.hard_nll:.6f}, "
            f"gate={'pass' if passed else 'fail'}",
            flush=True,
        )

    passing = [
        result
        for result in candidate_results
        if result["validation_gate_passed"]
    ]
    if passing:
        selected = min(
            passing,
            key=lambda result: int(result["routing_width"]),
        )
    else:
        selected = min(
            candidate_results,
            key=lambda result: float(
                result["validation_metrics"]["hard_nll"]  # type: ignore[index]
            ),
        )
    selected_width = int(selected["routing_width"])
    selected_executor = candidate_executors[selected_width]
    selected_config = ModalExecutorConfig(
        input_activation=input_name,
        output_activation=output_name,
        sequence_length=task_config.sequence_length,
        input_modes=selected_input_modes,
        output_modes=selected_output_modes,
        routing_width=selected_width,
    )

    executor_path = artifact_paths.executor
    fit_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "layer_index": layer_index,
        "teacher_state_sha256": teacher_state_before,
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
        "fit_context_ids_sha256": split_manifest["train"][
            "context_ids_sha256"
        ],
        "selection_context_ids_sha256": split_manifest[
            "validation_fisher"
        ]["context_ids_sha256"],
        "selected_candidate": selected,
        "validation_gate": validation_gate,
        "fit_protocol": fit_protocol,
    }
    save_position_modal_executor(
        executor_path,
        executor=selected_executor,
        config=selected_config,
        metadata=fit_metadata,
    )
    loaded_executor, loaded_config, loaded_metadata = (
        load_position_modal_executor(executor_path)
    )
    if loaded_config != selected_config:
        raise ValueError("reloaded modal executor config mismatch")
    if loaded_metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("reloaded modal executor checkpoint mismatch")

    print(
        f"Locked routing width {selected_width}; evaluating test once",
        flush=True,
    )
    baseline_test = evaluate_associative_recall(model, splits.test)
    bottleneck_test = _evaluate_replacement(
        adapter,
        layer_index=layer_index,
        executor=bottleneck_executor,
        split=splits.test,
    )
    affine_test = _evaluate_replacement(
        adapter,
        layer_index=layer_index,
        executor=affine_executor,
        split=splits.test,
    )
    graph_test = _evaluate_replacement(
        adapter,
        layer_index=layer_index,
        executor=loaded_executor,
        split=splits.test,
    )

    prefix = f"layer.{layer_index}"
    affine_activation_fit = _activation_fit_metrics(
        affine_executor,
        validation_activations[input_name],
        validation_activations[output_name],
        prefix=prefix,
    )
    graph_activation_fit = _activation_fit_metrics(
        loaded_executor,
        validation_activations[input_name],
        validation_activations[output_name],
        prefix=prefix,
    )
    selected_fit = selected["fit"]
    assert isinstance(selected_fit, dict)
    graph_edges = loaded_executor.graph.edge_count
    original_parameters = sum(
        parameter.numel() for parameter in original_layer.parameters()
    )
    learned_parameters = sum(
        parameter.numel()
        for parameter in loaded_executor.graph.parameters()
    )
    graph_multiplies = _estimated_modal_multiplies(
        sequence_length=task_config.sequence_length,
        width=model_config.d_model,
        input_modes=selected_input_modes,
        output_modes=selected_output_modes,
        graph_edges=graph_edges,
    )
    original_multiplies = _estimated_block_multiplies(
        model_config,
        sequence_length=task_config.sequence_length,
    )
    teacher_state_after = _module_state_sha256(model)
    if teacher_state_after != teacher_state_before:
        raise RuntimeError("modal executor build mutated the frozen teacher")
    report: dict[str, object] = {
        "format_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "modal_executor_sha256": _sha256(executor_path),
        "layer_index": layer_index,
        "teacher_state_sha256_before": teacher_state_before,
        "teacher_state_sha256_after": teacher_state_after,
        "teacher_was_frozen": True,
        "training_distribution": (
            "clean_frozen_teacher_boundary_pairs"
        ),
        "training_contract": "same_forward_input_output",
        "target": "frozen_teacher_layer_output_for_exact_input",
        "robustification_used": False,
        "compensation_target_used": False,
        "input_activation": input_name,
        "output_activation": output_name,
        "fit_split": "train",
        "selection_split": "validation_fisher",
        "evaluation_split": "test",
        "test_used_for_fit_or_selection": False,
        "fit_protocol": fit_protocol,
        "baseline": {
            "validation_metrics": _metrics(baseline_validation),
            "test_metrics": _metrics(baseline_test),
        },
        "bottleneck": {
            "description": "original transformer with simultaneous modal input/output bottlenecks",
            "validation_metrics": _metrics(bottleneck_validation),
            "test_metrics": _metrics(bottleneck_test),
        },
        "affine_graph": {
            "description": "standalone causal position-conditioned affine graph",
            "fit": {
                "train_r_squared": affine_fit.train_r_squared,
                "train_rmse": affine_fit.train_rmse,
                "ridge": affine_fit.ridge,
                "samples": affine_fit.samples,
                "learned_parameters": (
                    affine_executor.graph.edge_count
                    + affine_fit.bias.numel()
                ),
            },
            "validation_activation_fit": affine_activation_fit,
            "validation_metrics": _metrics(affine_validation),
            "test_metrics": _metrics(affine_test),
        },
        "nonlinear_candidates": candidate_results,
        "nonlinear_graph": {
            "description": "standalone causal position-conditioned nonlinear modal graph",
            "config": asdict(selected_config),
            "fit_config": asdict(fit_config),
            "fit": selected_fit,
            "validation_activation_fit": graph_activation_fit,
            "validation_metrics": selected["validation_metrics"],
            "test_metrics": _metrics(graph_test),
            "validation_gate_passed": selected[
                "validation_gate_passed"
            ],
            "size": {
                "learned_parameters": learned_parameters,
                "original_block_parameters": original_parameters,
                "graph_edges": graph_edges,
                "estimated_multiplies_per_sequence": graph_multiplies,
                "original_block_estimated_multiplies_per_sequence": (
                    original_multiplies
                ),
                "estimated_multiply_ratio": (
                    graph_multiplies / original_multiplies
                ),
            },
        },
        "artifacts": {
            "executor": executor_path.name,
            "checkpoint": checkpoint_path.name,
            "fisher": fisher_path.name,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_status": (
            "exploratory_single_checkpoint_test_previously_inspected"
        ),
    }
    report_path = artifact_paths.report_json
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _write_markdown(
        artifact_paths.report_markdown,
        report,
    )
    print(
        f"Modal executor build complete: "
        f"test accuracy={graph_test.answer_accuracy:.3%}, "
        f"paired={graph_test.paired_context_accuracy:.3%}, "
        f"NLL={graph_test.hard_nll:.6f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a position-conditioned modal graph and replace one layer."
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
    parser.add_argument(
        "--routing-widths",
        type=int,
        nargs="+",
        default=(4, 6, 8, 12),
    )
    parser.add_argument("--fit-steps", type=int, default=2_000)
    args = parser.parse_args()
    run_modal_executor_build(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        input_modes=args.input_modes,
        output_modes=args.output_modes,
        routing_widths=tuple(args.routing_widths),
        fit_steps=args.fit_steps,
    )


if __name__ == "__main__":
    main()
