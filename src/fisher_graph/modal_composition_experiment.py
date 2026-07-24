"""Compose independently compiled modal layers without teacher retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Mapping

import torch
from torch import Tensor

from .associative import (
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from .adapters import ToyTransformerAdapter, as_model_adapter
from .config import TransformerConfig
from .layers import LayerExecutor, TransformerBlock
from .modal_artifacts import (
    modal_completion_artifact_paths,
    modal_executor_artifact_paths,
)
from .modal_completion import (
    PositionConditionedCompletedModalGraphExecutor,
    load_position_modal_completion,
)
from .modal_executor import (
    ModalExecutorConfig,
    PositionConditionedModalGraphExecutor,
    load_position_modal_executor,
)
from .modal_executor_experiment import (
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


def _tensor_sha256(tensor: Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _replaced_layers(
    model: ToyTransformer | ToyTransformerAdapter,
    replacements: Mapping[int, LayerExecutor],
) -> Iterator[None]:
    adapter = as_model_adapter(model)
    segment_replacements: dict[str, LayerExecutor] = {}
    for index, executor in replacements.items():
        try:
            segment = adapter.segments[index]
        except IndexError:
            raise ValueError(
                "replacement index is outside the adapter segments"
            ) from None
        segment_replacements[segment.id] = executor
    with adapter.replaced_segments(segment_replacements):
        yield


def _system_logits(
    model: ToyTransformer | ToyTransformerAdapter,
    split,
    replacements: Mapping[int, LayerExecutor],
) -> Tensor:
    adapter = as_model_adapter(model)
    if not isinstance(adapter.module, ToyTransformer):
        raise TypeError(
            "the associative-recall evaluator requires a ToyTransformer"
        )
    with _replaced_layers(adapter, replacements):
        return associative_recall_answer_logits(adapter.module, split)


def _system_activations(
    model: ToyTransformer | ToyTransformerAdapter,
    split,
    replacements: Mapping[int, LayerExecutor],
) -> dict[str, Tensor]:
    adapter = as_model_adapter(model)
    with _replaced_layers(adapter, replacements):
        return _collect_activations(
            adapter,
            split,
            (
                "layer.0.output",
                "layer.1.input",
                "layer.1.output",
            ),
        )


def _behavior(
    split,
    logits: Tensor,
    reference_logits: Tensor,
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


def _cosine(
    left: Tensor,
    right: Tensor,
    matrix: Tensor | None = None,
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
    denominator = (left_norm.clamp_min(0) * right_norm.clamp_min(0)).sqrt()
    if denominator <= torch.finfo(torch.float64).eps:
        return 0.0
    return (numerator / denominator).item()


def _error_metrics(
    error: Tensor,
    fisher_matrix: Tensor,
) -> dict[str, object]:
    values = error.to(torch.float64)
    fisher = fisher_matrix.to(torch.float64)
    fisher_energy = torch.einsum(
        "...i,ij,...j->...",
        values,
        fisher,
        values,
    ).clamp_min(0)
    per_position_raw = values.square().mean(dim=(0, 2)).sqrt()
    per_position_fisher = fisher_energy.mean(dim=0).sqrt()
    return {
        "raw_rmse": values.square().mean().sqrt().item(),
        "raw_l2_per_token": (
            values.square().sum(dim=-1).mean().sqrt().item()
        ),
        "fisher_rms": fisher_energy.mean().sqrt().item(),
        "per_position_raw_rmse": per_position_raw.tolist(),
        "per_position_fisher_rms": per_position_fisher.tolist(),
    }


def _error_decomposition(
    teacher_output: Tensor,
    upstream_output: Tensor,
    composed_output: Tensor,
    fisher_matrix: Tensor,
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
        "upstream": _error_metrics(upstream_error, fisher_matrix),
        "local_same_input": _error_metrics(local_error, fisher_matrix),
        "total": _error_metrics(total_error, fisher_matrix),
        "upstream_local_raw_cosine": _cosine(
            upstream_error,
            local_error,
        ),
        "upstream_local_fisher_cosine": _cosine(
            upstream_error,
            local_error,
            fisher_matrix,
        ),
        "maximum_additive_identity_residual": (
            identity_residual.abs().max().item()
        ),
    }


def _boundary_identity(
    activations: Mapping[str, Tensor],
) -> dict[str, object]:
    difference = (
        activations["layer.0.output"]
        - activations["layer.1.input"]
    )
    return {
        "exactly_equal": bool(torch.equal(
            activations["layer.0.output"],
            activations["layer.1.input"],
        )),
        "maximum_absolute_difference": difference.abs().max().item(),
    }


def _validate_layer_artifacts(
    *,
    layer_index: int,
    checkpoint_hash: str,
    fisher_hash: str,
    executor_path: Path,
    output_completion_path: Path,
    expected_input: str,
    expected_output: str,
    sequence_length: int,
    width: int,
    teacher_state_sha256: str,
    fit_context_ids_sha256: str,
    selection_context_ids_sha256: str,
) -> tuple[
    PositionConditionedModalGraphExecutor,
    PositionConditionedCompletedModalGraphExecutor,
    ModalExecutorConfig,
    dict[str, object],
]:
    executor, config, executor_metadata = load_position_modal_executor(
        executor_path
    )
    completion, completion_config, completion_metadata = (
        load_position_modal_completion(output_completion_path)
    )
    if config.input_activation != expected_input:
        raise ValueError(
            f"layer {layer_index} executor input boundary mismatch"
        )
    if config.output_activation != expected_output:
        raise ValueError(
            f"layer {layer_index} executor output boundary mismatch"
        )
    if config.sequence_length != sequence_length:
        raise ValueError(
            f"layer {layer_index} executor sequence length mismatch"
        )
    if (
        completion_config.activation_name != expected_output
        or completion_config.sequence_length != sequence_length
        or completion_config.width != width
        or completion_config.kept_modes != config.output_modes
    ):
        raise ValueError(
            f"layer {layer_index} output completion boundary mismatch"
        )
    executor_hash = _sha256(executor_path)
    for label, metadata in (
        ("executor", executor_metadata),
        ("output completion", completion_metadata),
    ):
        if metadata.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(
                f"layer {layer_index} {label} checkpoint mismatch"
            )
        if metadata.get("fisher_sha256") != fisher_hash:
            raise ValueError(
                f"layer {layer_index} {label} Fisher mismatch"
            )
        if metadata.get("layer_index") != layer_index:
            raise ValueError(
                f"layer {layer_index} {label} index mismatch"
            )
        if metadata.get("teacher_state_sha256") != teacher_state_sha256:
            raise ValueError(
                f"layer {layer_index} {label} teacher state mismatch"
            )
    if completion_metadata.get("modal_executor_sha256") != executor_hash:
        raise ValueError(
            f"layer {layer_index} completion executor hash mismatch"
        )
    executor_contract = {
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
    for name, expected in executor_contract.items():
        if executor_metadata.get(name) != expected:
            raise ValueError(
                f"layer {layer_index} executor {name} mismatch"
            )
    if (
        completion_metadata.get("boundary_role") != "output"
        or completion_metadata.get("fit_activation") != expected_output
        or completion_metadata.get("training_distribution")
        != "clean_frozen_teacher_output"
    ):
        raise ValueError(
            f"layer {layer_index} completion provenance mismatch"
        )
    if (
        completion_metadata.get("fit_context_ids_sha256")
        != fit_context_ids_sha256
        or completion_metadata.get("selection_context_ids_sha256")
        != selection_context_ids_sha256
    ):
        raise ValueError(
            f"layer {layer_index} completion split provenance mismatch"
        )
    completion_protocol = completion_metadata.get("fit_protocol")
    if not isinstance(completion_protocol, dict):
        raise ValueError(
            f"layer {layer_index} completion fit protocol is missing"
        )
    if (
        completion_protocol.get("fit_split") != "train"
        or completion_protocol.get("selection_split")
        != "validation_fisher"
        or completion_protocol.get("test_used_for_fit_or_selection")
        is not False
    ):
        raise ValueError(
            f"layer {layer_index} completion fit provenance mismatch"
        )
    completed = PositionConditionedCompletedModalGraphExecutor(
        executor,
        completion,
    )
    if any(
        isinstance(module, TransformerBlock)
        for module in completed.modules()
    ):
        raise ValueError(
            f"layer {layer_index} compiled executor contains a transformer"
        )
    provenance = {
        "executor_sha256": executor_hash,
        "output_completion_sha256": _sha256(output_completion_path),
        "input_activation": expected_input,
        "output_activation": expected_output,
        "input_modes": config.input_modes,
        "output_modes": config.output_modes,
        "routing_width": config.routing_width,
        "teacher_state_sha256": executor_metadata[
            "teacher_state_sha256"
        ],
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
    return executor, completed, config, provenance


def _evaluate_split(
    *,
    model: ToyTransformer,
    split,
    systems: Mapping[str, Mapping[int, LayerExecutor]],
    output_basis: FisherModeBasis,
) -> dict[str, object]:
    logits = {
        name: _system_logits(model, split, replacements)
        for name, replacements in systems.items()
    }
    teacher_logits = logits["teacher"]
    behavior = {
        name: _behavior(split, values, teacher_logits)
        for name, values in logits.items()
    }
    activations = {
        name: _system_activations(
            model,
            split,
            systems[name],
        )
        for name in (
            "teacher",
            "layer_0_completed",
            "layer_1_completed",
            "both_completed",
        )
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
                "suffix_behavior": _behavior(
                    split,
                    logits["layer_1_completed"],
                    logits["teacher"],
                ),
                "layer_output_error": _error_metrics(
                    clean_local_error,
                    output_basis.matrix,
                ),
            },
            "compiled_layer_0_input": {
                "reference_system": "layer_0_completed",
                "compiled_system": "both_completed",
                "suffix_behavior": _behavior(
                    split,
                    logits["both_completed"],
                    logits["layer_0_completed"],
                ),
                "layer_output_error": _error_metrics(
                    shifted_local_error,
                    output_basis.matrix,
                ),
            },
        },
        "error_decomposition": _error_decomposition(
            activations["teacher"]["layer.1.output"],
            activations["layer_0_completed"]["layer.1.output"],
            activations["both_completed"]["layer.1.output"],
            output_basis.matrix,
        ),
        "boundary_identity": {
            name: _boundary_identity(values)
            for name, values in activations.items()
        },
    }


def _passes_validation_gate(
    evaluation: Mapping[str, object],
    gate: Mapping[str, float],
) -> bool:
    systems = evaluation["systems_vs_teacher"]
    contracts = evaluation["same_input_contracts"]
    assert isinstance(systems, dict)
    assert isinstance(contracts, dict)
    for name in ("layer_1_completed", "both_completed"):
        behavior = systems[name]
        assert isinstance(behavior, dict)
        metrics = behavior["metrics"]
        assert isinstance(metrics, dict)
        if (
            float(metrics["answer_accuracy"])
            < gate["minimum_answer_accuracy"]
            or float(metrics["paired_context_accuracy"])
            < gate["minimum_paired_accuracy"]
        ):
            return False
    for name in ("clean_input", "compiled_layer_0_input"):
        contract = contracts[name]
        assert isinstance(contract, dict)
        suffix = contract["suffix_behavior"]
        assert isinstance(suffix, dict)
        metrics = suffix["metrics"]
        assert isinstance(metrics, dict)
        reference_name = str(contract["reference_system"])
        reference = systems[reference_name]
        assert isinstance(reference, dict)
        reference_metrics = reference["metrics"]
        assert isinstance(reference_metrics, dict)
        if (
            float(metrics["hard_nll"])
            > float(reference_metrics["hard_nll"])
            + gate["maximum_same_input_nll_increase"]
            or float(suffix["reference_to_system_answer_kl"])
            > gate["maximum_same_input_answer_kl"]
        ):
            return False
    return True


def _layer_accounting(
    *,
    executor: PositionConditionedModalGraphExecutor,
    completed: PositionConditionedCompletedModalGraphExecutor,
    config: ModalExecutorConfig,
    sequence_length: int,
    width: int,
) -> dict[str, object]:
    base_multiplies = _estimated_modal_multiplies(
        sequence_length=sequence_length,
        width=width,
        input_modes=config.input_modes,
        output_modes=config.output_modes,
        graph_edges=executor.graph.edge_count,
    )
    output_completion = completed.output_completion
    parameter_tensors = list(completed.parameters())
    buffer_tensors = list(completed.buffers())
    parameter_elements = sum(
        tensor.numel() for tensor in parameter_tensors
    )
    buffer_elements = sum(tensor.numel() for tensor in buffer_tensors)
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
        "input_modes": config.input_modes,
        "routing_width": config.routing_width,
        "output_modes": config.output_modes,
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


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    validation = report["validation"]
    test = report["test"]
    accounting = report["accounting"]
    assert isinstance(validation, dict)
    assert isinstance(test, dict)
    assert isinstance(accounting, dict)
    lines = [
        "# Frozen Two-Layer Modal Composition",
        "",
        "Both transformer blocks were replaced by independently fitted modal",
        "graph executors. No transformer weight was updated, and layer 1 was",
        "trained only against its frozen teacher on matching clean inputs.",
        "",
        "## Behavior",
        "",
        "| Split | System | Answer accuracy | Paired accuracy | "
        "Hard NLL | KL vs teacher |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name, section in (
        ("Validation", validation),
        ("Exploratory test", test),
    ):
        systems = section["systems_vs_teacher"]
        assert isinstance(systems, dict)
        for name in (
            "teacher",
            "layer_0_completed",
            "layer_1_completed",
            "both_completed",
        ):
            behavior = systems[name]
            assert isinstance(behavior, dict)
            metrics = behavior["metrics"]
            assert isinstance(metrics, dict)
            lines.append(
                f"| {split_name} | {name.replace('_', ' ')} | "
                f"{float(metrics['answer_accuracy']):.3%} | "
                f"{float(metrics['paired_context_accuracy']):.3%} | "
                f"{float(metrics['hard_nll']):.6f} | "
                f"{float(behavior['reference_to_system_answer_kl']):.6f} |"
            )
    validation_contracts = validation["same_input_contracts"]
    validation_decomposition = validation["error_decomposition"]
    assert isinstance(validation_contracts, dict)
    assert isinstance(validation_decomposition, dict)
    shifted = validation_contracts["compiled_layer_0_input"]
    assert isinstance(shifted, dict)
    shifted_suffix = shifted["suffix_behavior"]
    assert isinstance(shifted_suffix, dict)
    local = validation_decomposition["local_same_input"]
    assert isinstance(local, dict)
    lines.extend(
        [
            "",
            "## Same-input composition contract",
            "",
            "The critical comparison holds the input to layer 1 fixed:",
            "`B1(E0(h))` versus `E1(E0(h))`. On validation its suffix KL is "
            f"{float(shifted_suffix['reference_to_system_answer_kl']):.6f}, "
            "and its Fisher-weighted layer-output RMS error is "
            f"{float(local['fisher_rms']):.6f}. This avoids mistaking",
            "downstream cancellation of layer-0 error for layer-1 fidelity.",
            "",
            "The raw upstream/local error cosine is "
            f"{float(validation_decomposition['upstream_local_raw_cosine']):.6f}; "
            "the Fisher-weighted cosine is "
            f"{float(validation_decomposition['upstream_local_fisher_cosine']):.6f}.",
            "",
            "## Compute estimate",
            "",
            f"- Both original blocks: "
            f"{accounting['original_two_block_estimated_multiplies']} multiplies",
            f"- Both completed modal graphs: "
            f"{accounting['compiled_two_layer_estimated_multiplies']} multiplies",
            f"- Compiled/original ratio: "
            f"{float(accounting['compiled_multiply_ratio']):.3%}",
            "",
            "These are block-only scalar-multiply estimates. Learned",
            "parameters and stored buffers are reported separately in JSON.",
            "",
            "This remains an exploratory single-checkpoint result because the",
            "test split was inspected during earlier development.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_modal_composition(
    *,
    artifact_dir: Path,
) -> dict[str, object]:
    """Load, compose, validation-gate, and test two frozen modal executors."""

    started = time.perf_counter()
    checkpoint_path = artifact_dir / "checkpoint.pt"
    fisher_path = artifact_dir / "fisher_modes.pt"
    manifest_path = artifact_dir / "split_manifest.json"
    report_path = artifact_dir / "modal_composition_report.json"
    markdown_path = artifact_dir / "modal_composition_report.md"
    checkpoint_hash = _sha256(checkpoint_path)
    fisher_hash = _sha256(fisher_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = TransformerConfig(**checkpoint["model_config"])
    if model_config.n_layers != 2:
        raise ValueError("the composition experiment requires two layers")
    model = ToyTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    teacher_state_before = _module_state_sha256(model)
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    splits = build_associative_recall_splits(task_config)
    manifest = json.loads(manifest_path.read_text())
    for split_name, split in (
        ("train", splits.train),
        ("validation_fisher", splits.validation),
        ("test", splits.test),
    ):
        manifest_split = manifest[split_name]
        if manifest_split["context_ids"] != split.context_ids.tolist():
            raise ValueError(
                f"split manifest context IDs mismatch: {split_name}"
            )
        if (
            manifest_split["context_ids_sha256"]
            != _tensor_sha256(split.context_ids)
        ):
            raise ValueError(
                f"split manifest context hash mismatch: {split_name}"
            )
    bases, _, _, fisher_metadata = load_fisher_build(fisher_path)
    if fisher_metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Fisher artifact does not match the checkpoint")

    executor_paths = {
        index: modal_executor_artifact_paths(artifact_dir, index)
        for index in (0, 1)
    }
    completion_paths = {
        index: modal_completion_artifact_paths(artifact_dir, index)
        for index in (0, 1)
    }
    runtime_paths = {
        "layer_0_executor": executor_paths[0].executor,
        "layer_0_output_completion": (
            completion_paths[0].output_completion
        ),
        "layer_1_executor": executor_paths[1].executor,
        "layer_1_output_completion": (
            completion_paths[1].output_completion
        ),
    }
    hashes_before_validation = {
        name: _sha256(path) for name, path in runtime_paths.items()
    }
    layer0, completed0, config0, provenance0 = (
        _validate_layer_artifacts(
            layer_index=0,
            checkpoint_hash=checkpoint_hash,
            fisher_hash=fisher_hash,
            executor_path=executor_paths[0].executor,
            output_completion_path=(
                completion_paths[0].output_completion
            ),
            expected_input="layer.0.input",
            expected_output="layer.0.output",
            sequence_length=task_config.sequence_length,
            width=model_config.d_model,
            teacher_state_sha256=teacher_state_before,
            fit_context_ids_sha256=manifest["train"][
                "context_ids_sha256"
            ],
            selection_context_ids_sha256=manifest[
                "validation_fisher"
            ]["context_ids_sha256"],
        )
    )
    layer1, completed1, config1, provenance1 = (
        _validate_layer_artifacts(
            layer_index=1,
            checkpoint_hash=checkpoint_hash,
            fisher_hash=fisher_hash,
            executor_path=executor_paths[1].executor,
            output_completion_path=(
                completion_paths[1].output_completion
            ),
            expected_input="layer.0.output",
            expected_output="layer.1.output",
            sequence_length=task_config.sequence_length,
            width=model_config.d_model,
            teacher_state_sha256=teacher_state_before,
            fit_context_ids_sha256=manifest["train"][
                "context_ids_sha256"
            ],
            selection_context_ids_sha256=manifest[
                "validation_fisher"
            ]["context_ids_sha256"],
        )
    )
    systems: dict[str, dict[int, LayerExecutor]] = {
        "teacher": {},
        "layer_0_zero_tail": {0: layer0},
        "layer_1_zero_tail": {1: layer1},
        "both_zero_tail": {0: layer0, 1: layer1},
        "layer_0_completed": {0: completed0},
        "layer_1_completed": {1: completed1},
        "both_completed": {0: completed0, 1: completed1},
        "layer_0_completed_layer_1_zero_tail": {
            0: completed0,
            1: layer1,
        },
        "layer_0_zero_tail_layer_1_completed": {
            0: layer0,
            1: completed1,
        },
    }

    print("Evaluating frozen two-layer composition on validation", flush=True)
    validation = _evaluate_split(
        model=model,
        split=splits.validation,
        systems=systems,
        output_basis=bases["layer.1.output"],
    )
    validation_gate = {
        "minimum_answer_accuracy": 0.995,
        "minimum_paired_accuracy": 0.99,
        "maximum_same_input_nll_increase": 0.01,
        "maximum_same_input_answer_kl": 0.01,
    }
    validation_gate_passed = _passes_validation_gate(
        validation,
        validation_gate,
    )
    if not validation_gate_passed:
        raise RuntimeError(
            "locked pristine executors failed the composition validation gate"
        )
    if any(
        not bool(item["exactly_equal"])
        for item in validation["boundary_identity"].values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("layer boundary alias identity failed")
    if {
        name: _sha256(path) for name, path in runtime_paths.items()
    } != hashes_before_validation:
        raise RuntimeError("runtime artifact changed during validation")
    if _module_state_sha256(model) != teacher_state_before:
        raise RuntimeError("teacher changed during composition validation")

    print("Composition locked; evaluating exploratory test once", flush=True)
    test = _evaluate_split(
        model=model,
        split=splits.test,
        systems=systems,
        output_basis=bases["layer.1.output"],
    )
    hashes_after_test = {
        name: _sha256(path) for name, path in runtime_paths.items()
    }
    teacher_state_after = _module_state_sha256(model)
    if hashes_after_test != hashes_before_validation:
        raise RuntimeError("runtime artifact changed after test lock")
    if teacher_state_after != teacher_state_before:
        raise RuntimeError("teacher changed during composition test")

    layer0_accounting = _layer_accounting(
        executor=layer0,
        completed=completed0,
        config=config0,
        sequence_length=task_config.sequence_length,
        width=model_config.d_model,
    )
    layer1_accounting = _layer_accounting(
        executor=layer1,
        completed=completed1,
        config=config1,
        sequence_length=task_config.sequence_length,
        width=model_config.d_model,
    )
    original_block_multiplies = _estimated_block_multiplies(
        model_config,
        sequence_length=task_config.sequence_length,
    )
    compiled_multiplies = (
        int(layer0_accounting["completed_estimated_multiplies"])
        + int(layer1_accounting["completed_estimated_multiplies"])
    )
    compiled_parameter_elements = sum(
        int(layer["storage"]["learned_parameter_elements"])  # type: ignore[index]
        for layer in (layer0_accounting, layer1_accounting)
    )
    compiled_parameter_bytes = sum(
        int(layer["storage"]["learned_parameter_bytes"])  # type: ignore[index]
        for layer in (layer0_accounting, layer1_accounting)
    )
    compiled_buffer_elements = sum(
        int(layer["storage"]["stored_buffer_elements"])  # type: ignore[index]
        for layer in (layer0_accounting, layer1_accounting)
    )
    compiled_buffer_bytes = sum(
        int(layer["storage"]["stored_buffer_bytes"])  # type: ignore[index]
        for layer in (layer0_accounting, layer1_accounting)
    )
    report: dict[str, object] = {
        "format_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "teacher_state_sha256_before": teacher_state_before,
        "teacher_state_sha256_after": teacher_state_after,
        "teacher_was_frozen": True,
        "protocol": {
            "layer_0_training_distribution": (
                provenance0["training_distribution"]
            ),
            "layer_0_training_contract": provenance0[
                "training_contract"
            ],
            "layer_0_target": provenance0["target"],
            "layer_1_training_distribution": (
                provenance1["training_distribution"]
            ),
            "layer_1_training_contract": provenance1[
                "training_contract"
            ],
            "layer_1_target": provenance1["target"],
            "layer_1_robustification_used": provenance1[
                "robustification_used"
            ],
            "forbidden_compensation_pair_used": provenance1[
                "compensation_target_used"
            ],
            "validation_split": "validation_fisher",
            "evaluation_split": "test",
            "test_used_for_fit_or_selection": bool(
                provenance0["test_used_for_fit_or_selection"]
                or provenance1["test_used_for_fit_or_selection"]
            ),
            "validation_gate": validation_gate,
            "validation_gate_passed": validation_gate_passed,
        },
        "layer_provenance": {
            "layer_0": provenance0,
            "layer_1_pristine": provenance1,
        },
        "validation": validation,
        "test": test,
        "accounting": {
            "layer_0": layer0_accounting,
            "layer_1": layer1_accounting,
            "original_block_estimated_multiplies": (
                original_block_multiplies
            ),
            "original_two_block_estimated_multiplies": (
                2 * original_block_multiplies
            ),
            "compiled_two_layer_estimated_multiplies": (
                compiled_multiplies
            ),
            "compiled_multiply_ratio": (
                compiled_multiplies / (2 * original_block_multiplies)
            ),
            "compiled_learned_parameter_elements": (
                compiled_parameter_elements
            ),
            "compiled_learned_parameter_bytes": compiled_parameter_bytes,
            "compiled_stored_buffer_elements": (
                compiled_buffer_elements
            ),
            "compiled_stored_buffer_bytes": compiled_buffer_bytes,
        },
        "artifacts": {
            name: {
                "filename": runtime_paths[name].name,
                "sha256": hashes_before_validation[name],
            }
            for name in runtime_paths
        },
        "artifact_hashes_locked_before_validation_and_test": True,
        "scientific_status": (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _write_markdown(markdown_path, report)
    both_test = test["systems_vs_teacher"]["both_completed"]  # type: ignore[index]
    both_test_metrics = both_test["metrics"]  # type: ignore[index]
    print(
        "Two-layer composition complete: "
        f"test accuracy={float(both_test_metrics['answer_accuracy']):.3%}, "
        f"paired={float(both_test_metrics['paired_context_accuracy']):.3%}, "
        f"NLL={float(both_test_metrics['hard_nll']):.6f}, "
        f"compute={compiled_multiplies}/"
        f"{2 * original_block_multiplies}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose two independently trained frozen modal layer executors."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/associative_recall"),
    )
    args = parser.parse_args()
    run_modal_composition(artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
