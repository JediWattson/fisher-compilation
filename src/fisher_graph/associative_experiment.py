"""End-to-end associative-recall training and Fisher-mode build."""

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
    AssociativeRecallTrainingConfig,
    associative_recall_model_config,
    run_associative_recall_experiment,
)
from .adapters import ToyTransformerAdapter
from .compiler.calibration import CalibrationBatch
from .modes import (
    ActivationGradientSamples,
    FisherModeBasis,
    ModalJacobian,
    ModalTransition,
    build_fisher_mode_bases,
    collect_activation_score_gradients,
    extract_segment_modal_jacobian,
    fit_modal_transition,
    save_fisher_build,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _metrics_dict(metrics: AssociativeRecallMetrics) -> dict[str, object]:
    return asdict(metrics)


def _basis_diagnostics(
    basis: FisherModeBasis,
    samples: ActivationGradientSamples,
) -> dict[str, object]:
    reconstructed = (
        basis.vectors
        @ torch.diag(basis.eigenvalues)
        @ basis.vectors.transpose(0, 1)
    )
    fisher_norm = basis.matrix.norm().clamp_min(torch.finfo(torch.float64).eps)
    reconstruction_error = (
        (reconstructed - basis.matrix).norm() / fisher_norm
    ).item()
    identity = torch.eye(basis.width, dtype=basis.vectors.dtype)
    orthogonality_error = (
        basis.vectors.transpose(0, 1) @ basis.vectors - identity
    ).abs().max().item()
    expected_trace = (
        samples.score_gradients.to(torch.float64).square().sum()
        / samples.observations
    )
    trace_error = (
        (basis.eigenvalues.sum() - expected_trace).abs()
        / expected_trace.abs().clamp_min(torch.finfo(torch.float64).eps)
    ).item()
    positive = basis.eigenvalues[basis.eigenvalues > 0]
    probabilities = positive / positive.sum() if positive.numel() else positive
    effective_rank = (
        (-(probabilities * probabilities.log()).sum()).exp().item()
        if probabilities.numel()
        else 0.0
    )
    finite = (
        torch.isfinite(basis.matrix).all()
        and torch.isfinite(basis.eigenvalues).all()
        and torch.isfinite(basis.vectors).all()
    )
    validation_passed = bool(
        finite
        and basis.eigenvalues.min() >= 0
        and orthogonality_error < 1e-10
        and reconstruction_error < 1e-10
        and trace_error < 1e-10
    )
    return {
        "width": basis.width,
        "sequences": basis.sequences,
        "observations": basis.observations,
        "scope": basis.scope,
        "score_reduction": basis.score_reduction,
        "normalizer": basis.normalizer,
        "fisher_trace": basis.fisher_trace,
        "effective_rank": effective_rank,
        "top_eigenvalues": basis.eigenvalues[:10].tolist(),
        "retained_curve": basis.retained_curve.tolist(),
        "modes_for_90_percent": basis.modes_for_fraction(0.90),
        "modes_for_95_percent": basis.modes_for_fraction(0.95),
        "modes_for_99_percent": basis.modes_for_fraction(0.99),
        "orthogonality_max_error": orthogonality_error,
        "fisher_reconstruction_relative_error": reconstruction_error,
        "trace_relative_error": trace_error,
        "finite": bool(finite),
        "validation_passed": validation_passed,
    }


def _transition_dict(transition: ModalTransition) -> dict[str, object]:
    return {
        "input_activation": transition.input_activation,
        "output_activation": transition.output_activation,
        "sequence_length": transition.sequence_length,
        "input_modes": transition.input_modes,
        "output_modes": transition.output_modes,
        "descriptive_fit_r_squared": transition.r_squared,
        "descriptive_fit_rmse": transition.rmse,
        "strongest_descriptive_edges": transition.strongest_edges(20),
    }


def _jacobian_dict(jacobian: ModalJacobian) -> dict[str, object]:
    return {
        "input_activation": jacobian.input_activation,
        "output_activation": jacobian.output_activation,
        "sequence_length": jacobian.sequence_length,
        "input_modes": jacobian.input_modes,
        "output_modes": jacobian.output_modes,
        "samples": jacobian.samples,
        "strongest_rms_edges": jacobian.strongest_edges(20),
    }


def _write_markdown_report(path: Path, report: dict[str, object]) -> None:
    training = report["training"]
    activations = report["activation_modes"]
    transitions = report["modal_transitions"]
    jacobians = report["modal_jacobians"]
    assert isinstance(training, dict)
    assert isinstance(activations, dict)
    assert isinstance(transitions, list)
    assert isinstance(jacobians, list)
    validation = training["validation"]
    test = training["test"]
    assert isinstance(validation, dict)
    assert isinstance(test, dict)

    lines = [
        "# Associative Recall Fisher Build",
        "",
        "## Trained model",
        "",
        f"- Selected checkpoint step: {training['best_step']}",
        f"- Validation answer accuracy: "
        f"{float(validation['answer_accuracy']):.3%}",
        f"- Validation paired-context accuracy: "
        f"{float(validation['paired_context_accuracy']):.3%}",
        f"- Test answer accuracy: {float(test['answer_accuracy']):.3%}",
        f"- Test paired-context accuracy: "
        f"{float(test['paired_context_accuracy']):.3%}",
        f"- Mean correct-answer probability: "
        f"{float(validation['mean_correct_probability']):.3%}",
        "",
        "## Width-pooled Fisher modes",
        "",
        "| Activation | Fisher trace | Effective rank | k90 | k95 | k99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in activations.items():
        assert isinstance(value, dict)
        lines.append(
            f"| `{name}` | {float(value['fisher_trace']):.6e} | "
            f"{float(value['effective_rank']):.3f} | "
            f"{value['modes_for_90_percent']} | "
            f"{value['modes_for_95_percent']} | "
            f"{value['modes_for_99_percent']} |"
        )

    lines.extend(
        [
            "",
            "Each basis diagonalizes the full 32 x 32 empirical Fisher",
            "constructed from hard-target, summed-NLL activation score",
            "gradients on the validation/Fisher split. Token positions are",
            "pooled as observations, producing modes reusable across positions.",
            "",
            "## Position-coupled modal computation",
            "",
            "| Layer | Modes in/out | Descriptive R2 | Jacobian samples |",
            "|---|---:|---:|---:|",
        ]
    )
    for index, (transition, jacobian) in enumerate(
        zip(transitions, jacobians, strict=True)
    ):
        assert isinstance(transition, dict)
        assert isinstance(jacobian, dict)
        lines.append(
            f"| {index} | {transition['input_modes']}/"
            f"{transition['output_modes']} | "
            f"{float(transition['descriptive_fit_r_squared']):.6f} | "
            f"{jacobian['samples']} |"
        )
    lines.extend(
        [
            "",
            "The saved modal Jacobians have axes",
            "`[output_position, output_mode, input_position, input_mode]`.",
            "Both signed means and RMS magnitudes are stored. The affine",
            "transitions are descriptive dataset fits and are not claimed as",
            "causal executors.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_build(
    *,
    output_dir: Path,
    task_config: AssociativeRecallTaskConfig,
    training_config: AssociativeRecallTrainingConfig,
    mode_fraction: float = 0.99,
    jacobian_modes: int = 8,
    jacobian_samples: int = 24,
) -> dict[str, object]:
    """Train associative recall and build validated Fisher-mode artifacts."""

    if not 0.0 < mode_fraction <= 1.0:
        raise ValueError("mode_fraction must be in (0, 1]")
    if jacobian_modes <= 0 or jacobian_samples <= 0:
        raise ValueError("Jacobian mode and sample counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Training deterministic associative-recall model", flush=True)
    result = run_associative_recall_experiment(
        task_config=task_config,
        training_config=training_config,
        model_config=associative_recall_model_config(task_config),
    )
    adapter = ToyTransformerAdapter(result.model)
    print(
        f"Selected step {result.best_checkpoint.step}: "
        f"validation={result.best_checkpoint.validation_metrics.answer_accuracy:.3%}, "
        f"test={result.test_metrics.answer_accuracy:.3%}",
        flush=True,
    )

    split_manifest = {
        "split_seed": task_config.split_seed,
        "train": {
            "contexts": result.splits.train.contexts,
            "examples": result.splits.train.samples,
            "context_ids": result.splits.train.context_ids.tolist(),
            "context_ids_sha256": _tensor_sha256(
                result.splits.train.context_ids
            ),
        },
        "validation_fisher": {
            "contexts": result.splits.validation.contexts,
            "examples": result.splits.validation.samples,
            "context_ids": result.splits.validation.context_ids.tolist(),
            "context_ids_sha256": _tensor_sha256(
                result.splits.validation.context_ids
            ),
        },
        "test": {
            "contexts": result.splits.test.contexts,
            "examples": result.splits.test.samples,
            "context_ids": result.splits.test.context_ids.tolist(),
            "context_ids_sha256": _tensor_sha256(
                result.splits.test.context_ids
            ),
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n")

    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_lines = [
        json.dumps(
            {
                "step": evaluation.step,
                "batch_training_loss": evaluation.batch_training_loss,
                "train": _metrics_dict(evaluation.train),
                "validation": _metrics_dict(evaluation.validation),
            }
        )
        for evaluation in result.history
    ]
    metrics_path.write_text("\n".join(metrics_lines) + "\n")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(result.model_config),
            "model_state_dict": result.best_checkpoint.model_state_dict,
            "task_config": asdict(task_config),
            "training_config": asdict(training_config),
            "best_step": result.best_checkpoint.step,
            "train_metrics": _metrics_dict(
                result.best_checkpoint.train_metrics
            ),
            "validation_metrics": _metrics_dict(
                result.best_checkpoint.validation_metrics
            ),
            "test_metrics": _metrics_dict(result.test_metrics),
            "split_context_ids": {
                "train": result.splits.train.context_ids,
                "validation_fisher": result.splits.validation.context_ids,
                "test": result.splits.test.context_ids,
            },
        },
        checkpoint_path,
    )
    checkpoint_hash = _file_sha256(checkpoint_path)

    tap_names = list(adapter.default_fisher_sites)
    if not tap_names:
        raise RuntimeError("model adapter did not select any Fisher sites")
    print(
        f"Collecting hard-target score gradients for "
        f"{result.splits.validation.samples} validation sequences",
        flush=True,
    )
    collection = collect_activation_score_gradients(
        adapter,
        result.splits.validation.input_ids,
        result.splits.validation.targets,
        activation_names=tap_names,
        ignore_index=task_config.ignore_index,
    )
    bases = build_fisher_mode_bases(collection)
    diagnostics = {
        name: _basis_diagnostics(bases[name], collection.samples[name])
        for name in tap_names
    }
    if not all(
        bool(value["validation_passed"]) for value in diagnostics.values()
    ):
        raise RuntimeError("one or more Fisher decompositions failed validation")

    segments = adapter.segments
    jacobian_calibration = CalibrationBatch(
        model_inputs={
            "input_ids": result.splits.validation.input_ids,
        },
        targets=result.splits.validation.targets,
        valid_positions=torch.ones_like(
            result.splits.validation.input_ids,
            dtype=torch.bool,
        ),
        example_ids=tuple(
            f"sequence.{index}"
            for index in range(result.splits.validation.samples)
        ),
    )
    transitions: list[ModalTransition] = []
    jacobians: list[ModalJacobian] = []
    for segment in segments:
        input_name = segment.input_site
        output_name = segment.output_site
        input_modes = bases[input_name].modes_for_fraction(mode_fraction)
        output_modes = bases[output_name].modes_for_fraction(mode_fraction)
        transitions.append(
            fit_modal_transition(
                collection.samples[input_name],
                collection.samples[output_name],
                bases[input_name],
                bases[output_name],
                input_modes=input_modes,
                output_modes=output_modes,
            )
        )
        diagnostic_input_modes = min(jacobian_modes, input_modes)
        diagnostic_output_modes = min(jacobian_modes, output_modes)
        print(
            f"Building segment {segment.id} modal Jacobian "
            f"({diagnostic_input_modes} input modes, "
            f"{diagnostic_output_modes} output modes, "
            f"{jacobian_samples} sequences)",
            flush=True,
        )
        jacobians.append(
            extract_segment_modal_jacobian(
                adapter,
                segment,
                (jacobian_calibration,),
                collection.samples[input_name],
                bases[input_name],
                bases[output_name],
                input_modes=diagnostic_input_modes,
                output_modes=diagnostic_output_modes,
                max_sequences=jacobian_samples,
            )
        )

    fisher_path = output_dir / "fisher_modes.pt"
    save_fisher_build(
        fisher_path,
        bases=bases,
        transitions=transitions,
        jacobians=jacobians,
        metadata={
            "task": "two_pair_associative_recall",
            "checkpoint_sha256": checkpoint_hash,
            "model_config": asdict(result.model_config),
            "validation_context_ids_sha256": split_manifest[
                "validation_fisher"
            ]["context_ids_sha256"],
            "estimator": {
                "kind": "empirical_activation_fisher",
                "scope": "width_pooled",
                "score": "hard_target_summed_negative_log_likelihood",
                "normalizer": "valid_activation_positions",
                "mode_fraction": mode_fraction,
            },
        },
    )

    report: dict[str, object] = {
        "task": {
            "name": "two_pair_associative_recall",
            "config": asdict(task_config),
            "splits": {
                name: {
                    "contexts": split["contexts"],
                    "examples": split["examples"],
                    "context_ids_sha256": split["context_ids_sha256"],
                }
                for name, split in split_manifest.items()
                if isinstance(split, dict)
            },
        },
        "model": asdict(result.model_config),
        "adapter": {
            "kind": type(adapter).__name__,
            "sequence": asdict(adapter.sequence_spec),
            "default_fisher_sites": tap_names,
            "segments": [
                asdict(segment) for segment in adapter.segments
            ],
        },
        "training": {
            "converged": result.converged,
            "final_step": result.final_step,
            "best_step": result.best_checkpoint.step,
            "label_smoothing": training_config.label_smoothing,
            "train": _metrics_dict(result.best_checkpoint.train_metrics),
            "validation": _metrics_dict(
                result.best_checkpoint.validation_metrics
            ),
            "test": _metrics_dict(result.test_metrics),
        },
        "fisher_estimator": {
            "dataset": "validation_fisher",
            "mean_sequence_hard_nll": collection.mean_loss,
            "scope": "width_pooled",
            "score_reduction": "sum",
            "normalizer": "valid_activation_positions",
            "mode_fraction": mode_fraction,
        },
        "activation_modes": diagnostics,
        "modal_transitions": [
            _transition_dict(transition) for transition in transitions
        ],
        "modal_jacobians": [
            _jacobian_dict(jacobian) for jacobian in jacobians
        ],
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "fisher_modes": fisher_path.name,
            "split_manifest": manifest_path.name,
            "training_metrics": metrics_path.name,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "fisher_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown_report(output_dir / "fisher_report.md", report)
    print(
        f"Fisher build complete in {report['elapsed_seconds']:.1f}s: "
        f"{fisher_path}",
        flush=True,
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train associative recall and build Fisher compute-mode artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/associative_recall"),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--mode-fraction", type=float, default=0.99)
    parser.add_argument("--jacobian-modes", type=int, default=8)
    parser.add_argument("--jacobian-samples", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_build(
        output_dir=args.output_dir,
        task_config=AssociativeRecallTaskConfig(),
        training_config=AssociativeRecallTrainingConfig(
            max_steps=args.max_steps,
            evaluation_interval=args.evaluation_interval,
            device=args.device,
        ),
        mode_fraction=args.mode_fraction,
        jacobian_modes=args.jacobian_modes,
        jacobian_samples=args.jacobian_samples,
    )


if __name__ == "__main__":
    main()
