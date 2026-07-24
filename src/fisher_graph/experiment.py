"""Train modular addition and build Fisher compute modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import TransformerConfig
from .modes import (
    FisherModeBasis,
    ModalJacobian,
    ModalTransition,
    build_fisher_mode_bases,
    collect_activation_score_gradients,
    extract_modal_jacobian,
    fit_modal_transition,
    save_fisher_build,
)
from .model import ToyTransformer
from .task import ModularAdditionTask
from .training import (
    TrainingConfig,
    save_checkpoint,
    train_modular_addition,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basis_report(basis: FisherModeBasis) -> dict[str, object]:
    identity = torch.eye(basis.width, dtype=basis.vectors.dtype)
    orthogonality_error = (
        basis.vectors.transpose(0, 1) @ basis.vectors - identity
    ).abs().max()
    return {
        "width": basis.width,
        "sequences": basis.sequences,
        "observations": basis.observations,
        "scope": basis.scope,
        "score_reduction": basis.score_reduction,
        "normalizer": basis.normalizer,
        "fisher_trace": basis.fisher_trace,
        "top_eigenvalues": basis.eigenvalues[:10].tolist(),
        "modes_for_90_percent": basis.modes_for_fraction(0.90),
        "modes_for_95_percent": basis.modes_for_fraction(0.95),
        "modes_for_99_percent": basis.modes_for_fraction(0.99),
        "orthogonality_max_error": orthogonality_error.item(),
    }


def _transition_report(transition: ModalTransition) -> dict[str, object]:
    return {
        "input_activation": transition.input_activation,
        "output_activation": transition.output_activation,
        "sequence_length": transition.sequence_length,
        "input_modes": transition.input_modes,
        "output_modes": transition.output_modes,
        "fit_r_squared": transition.r_squared,
        "fit_rmse": transition.rmse,
        "strongest_edges": transition.strongest_edges(20),
    }


def _jacobian_report(jacobian: ModalJacobian) -> dict[str, object]:
    return {
        "input_activation": jacobian.input_activation,
        "output_activation": jacobian.output_activation,
        "sequence_length": jacobian.sequence_length,
        "input_modes": jacobian.input_modes,
        "output_modes": jacobian.output_modes,
        "samples": jacobian.samples,
        "strongest_edges": jacobian.strongest_edges(20),
    }


def _markdown_report(report: dict[str, object]) -> str:
    training = report["training"]
    assert isinstance(training, dict)
    lines = [
        "# Modular Addition Fisher Build",
        "",
        "## Training",
        "",
        f"- Accuracy: {float(training['accuracy']):.3%}",
        f"- Loss: {float(training['loss']):.6f}",
        f"- Steps: {int(training['steps'])}",
        "",
        "## Fisher mode bases",
        "",
        "| Activation | Trace | 90% modes | 95% modes | 99% modes |",
        "|---|---:|---:|---:|---:|",
    ]
    activations = report["activations"]
    assert isinstance(activations, dict)
    for name, value in activations.items():
        assert isinstance(value, dict)
        lines.append(
            f"| `{name}` | {float(value['fisher_trace']):.6e} | "
            f"{int(value['modes_for_90_percent'])} | "
            f"{int(value['modes_for_95_percent'])} | "
            f"{int(value['modes_for_99_percent'])} |"
        )
    lines.extend(
        [
            "",
            "The estimator is the width-pooled empirical activation Fisher:",
            "valid token-position score gradients are outer-product averaged",
            "into one feature-space matrix per activation boundary.",
            "",
            "## Modal layer maps",
            "",
            "| Layer map | Modes in/out | Affine fit R2 | Affine fit RMSE | "
            "Jacobian samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    transitions = report["transitions"]
    jacobians = report["jacobians"]
    assert isinstance(transitions, list)
    assert isinstance(jacobians, list)
    for transition, jacobian in zip(transitions, jacobians, strict=True):
        assert isinstance(transition, dict)
        assert isinstance(jacobian, dict)
        lines.append(
            f"| `{transition['input_activation']}` -> "
            f"`{transition['output_activation']}` | "
            f"{transition['input_modes']}/{transition['output_modes']} | "
            f"{float(transition['fit_r_squared']):.6f} | "
            f"{float(transition['fit_rmse']):.6e} | "
            f"{jacobian['samples']} |"
        )
    lines.extend(
        [
            "",
            "The affine map is a dataset-local descriptive fit. The modal",
            "Jacobian is the local computation diagnostic and preserves all",
            "token-to-token blocks; its artifact stores both signed mean and",
            "RMS edge strength to expose context-dependent cancellation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    *,
    output_dir: Path,
    modulus: int,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    mode_fraction: float,
    jacobian_modes: int,
    jacobian_samples: int,
    verbose: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(training_config.seed)
    task = ModularAdditionTask(modulus=modulus)
    if model_config.vocab_size != task.vocab_size:
        raise ValueError("model vocabulary must match the modular-addition task")
    model = ToyTransformer(model_config)

    started = time.perf_counter()
    if verbose:
        print("Training modular-addition transformer", flush=True)
    training_result = train_modular_addition(
        model,
        task,
        training_config,
        verbose=verbose,
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        task=task,
        training_config=training_config,
        result=training_result,
    )
    checkpoint_hash = _sha256(checkpoint_path)

    inputs, targets = task.dataset()
    activation_names = [
        name
        for layer_index in range(model_config.n_layers)
        for name in (
            f"layer.{layer_index}.input",
            f"layer.{layer_index}.output",
        )
    ]
    if verbose:
        print("Collecting per-example activation score gradients", flush=True)
    collection = collect_activation_score_gradients(
        model,
        inputs,
        targets,
        activation_names=activation_names,
        ignore_index=task.ignore_index,
    )
    bases = build_fisher_mode_bases(collection)

    transitions: list[ModalTransition] = []
    jacobians: list[ModalJacobian] = []
    for layer_index in range(model_config.n_layers):
        input_name = f"layer.{layer_index}.input"
        output_name = f"layer.{layer_index}.output"
        input_modes = bases[input_name].modes_for_fraction(mode_fraction)
        output_modes = bases[output_name].modes_for_fraction(mode_fraction)
        transition = fit_modal_transition(
            collection.samples[input_name],
            collection.samples[output_name],
            bases[input_name],
            bases[output_name],
            input_modes=input_modes,
            output_modes=output_modes,
        )
        transitions.append(transition)

        diagnostic_input_modes = min(jacobian_modes, input_modes)
        diagnostic_output_modes = min(jacobian_modes, output_modes)
        if verbose:
            print(
                f"Extracting layer {layer_index} modal Jacobian "
                f"({diagnostic_input_modes}x{diagnostic_output_modes} modes, "
                f"{jacobian_samples} sequences)",
                flush=True,
            )
        jacobians.append(
            extract_modal_jacobian(
                model.layers[layer_index],
                collection.samples[input_name],
                bases[input_name],
                bases[output_name],
                input_modes=diagnostic_input_modes,
                output_modes=diagnostic_output_modes,
                max_sequences=jacobian_samples,
            )
        )

    artifact_path = output_dir / "fisher_modes.pt"
    metadata = {
        "task": "modular_addition",
        "modulus": modulus,
        "checkpoint_sha256": checkpoint_hash,
        "model_config": asdict(model_config),
        "estimator": {
            "kind": "empirical_activation_fisher",
            "scope": "width_pooled",
            "score": "summed_negative_log_likelihood",
            "normalizer": "valid_activation_positions",
            "mode_fraction": mode_fraction,
        },
    }
    save_fisher_build(
        artifact_path,
        bases=bases,
        transitions=transitions,
        jacobians=jacobians,
        metadata=metadata,
    )

    report: dict[str, object] = {
        "task": {
            "name": "modular_addition",
            "modulus": modulus,
            "examples": inputs.shape[0],
            "sequence_length": inputs.shape[1],
        },
        "model": asdict(model_config),
        "training": {
            "steps": training_result.steps,
            "loss": training_result.loss,
            "accuracy": training_result.accuracy,
            "history": list(training_result.history),
        },
        "fisher": {
            "mean_sequence_nll": collection.mean_loss,
            "scope": "width_pooled",
            "score_reduction": "sum",
            "normalizer": "valid_activation_positions",
            "mode_fraction": mode_fraction,
        },
        "activations": {
            name: _basis_report(basis) for name, basis in bases.items()
        },
        "transitions": [
            _transition_report(transition) for transition in transitions
        ],
        "jacobians": [
            _jacobian_report(jacobian) for jacobian in jacobians
        ],
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "fisher_modes": artifact_path.name,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(_markdown_report(report))
    if verbose:
        print(
            f"Build complete: {training_result.accuracy:.3%} accuracy, "
            f"artifacts in {output_dir}",
            flush=True,
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the toy transformer and build Fisher compute modes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/modular_addition"),
    )
    parser.add_argument("--modulus", type=int, default=13)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=2_500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--target-loss", type=float, default=0.08)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--mode-fraction", type=float, default=0.99)
    parser.add_argument("--jacobian-modes", type=int, default=8)
    parser.add_argument("--jacobian-samples", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task = ModularAdditionTask(modulus=args.modulus)
    model_config = TransformerConfig(
        vocab_size=task.vocab_size,
        max_sequence_length=task.sequence_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
    )
    run_experiment(
        output_dir=args.output_dir,
        modulus=args.modulus,
        model_config=model_config,
        training_config=training_config,
        mode_fraction=args.mode_fraction,
        jacobian_modes=args.jacobian_modes,
        jacobian_samples=args.jacobian_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
