"""Training and evaluation for variable-layout associative recall."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .activations import ActivationIntervention
from .config import TransformerConfig
from .model import ToyTransformer
from .variable_associative import (
    VariableAssociativeLayout,
    VariableAssociativeRecallSplit,
    VariableAssociativeRecallSplits,
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
    variable_associative_recall_model_config,
)


_CHECKPOINT_SCHEMA = "fisher_graph.variable_associative_checkpoint"
_CHECKPOINT_VERSION = 1
DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT = Path(
    ".local-runs/variable-associative/checkpoint.pt"
)


@dataclass(frozen=True, slots=True)
class VariableAssociativeTrainingConfig:
    """Deterministic optimizer and stopping configuration."""

    model_seed: int = 26_071
    shuffle_seed: int = 26_072
    learning_rate: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    batch_size: int = 256
    evaluation_batch_size: int = 512
    max_steps: int = 4_000
    evaluation_interval: int = 100
    gradient_clip_norm: float = 1.0
    minimum_answer_accuracy: float = 0.995
    minimum_paired_context_accuracy: float = 0.99
    minimum_stratum_accuracy: float = 0.99
    required_consecutive_evaluations: int = 2
    deterministic_algorithms: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.model_seed) is not int or type(self.shuffle_seed) is not int:
            raise ValueError("training seeds must be integers")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            len(self.betas) != 2
            or any(not 0 <= value < 1 for value in self.betas)
        ):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("epsilon must be positive and weight_decay nonnegative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        for name in (
            "batch_size",
            "evaluation_batch_size",
            "max_steps",
            "evaluation_interval",
            "required_consecutive_evaluations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        for name in (
            "minimum_answer_accuracy",
            "minimum_paired_context_accuracy",
            "minimum_stratum_accuracy",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")


@dataclass(frozen=True, slots=True)
class VariableAssociativeMetrics:
    """Behavior and worst-stratum metrics for one split."""

    samples: int
    contexts: int
    hard_nll: float
    answer_accuracy: float
    paired_context_accuracy: float
    query_accuracies: tuple[float, float]
    pair_order_accuracies: tuple[float, float]
    layout_accuracies: tuple[float, ...]
    length_accuracies: tuple[tuple[int, float], ...]
    minimum_query_accuracy: float
    minimum_pair_order_accuracy: float
    minimum_layout_accuracy: float
    minimum_length_accuracy: float
    mean_correct_probability: float

    @property
    def minimum_stratum_accuracy(self) -> float:
        return min(
            self.minimum_query_accuracy,
            self.minimum_pair_order_accuracy,
            self.minimum_layout_accuracy,
            self.minimum_length_accuracy,
        )


@dataclass(frozen=True, slots=True)
class VariableAssociativeEvaluation:
    step: int
    batch_training_loss: float
    train: VariableAssociativeMetrics
    validation: VariableAssociativeMetrics


@dataclass(frozen=True, slots=True)
class VariableAssociativeCheckpoint:
    step: int
    model_state_dict: dict[str, Tensor]
    train_metrics: VariableAssociativeMetrics
    validation_metrics: VariableAssociativeMetrics


@dataclass(slots=True)
class VariableAssociativeTrainingResult:
    model: ToyTransformer
    model_config: TransformerConfig
    task_config: VariableAssociativeRecallTaskConfig
    training_config: VariableAssociativeTrainingConfig
    splits: VariableAssociativeRecallSplits
    best_checkpoint: VariableAssociativeCheckpoint
    history: tuple[VariableAssociativeEvaluation, ...]
    final_step: int
    converged: bool
    test_metrics: VariableAssociativeMetrics


def _model_device(model: ToyTransformer) -> torch.device:
    reference = next(model.parameters(), None)
    if reference is None:
        raise ValueError("model must contain parameters")
    return reference.device


@torch.no_grad()
def variable_associative_answer_logits(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    batch_size: int = 512,
    activation_interventions: (
        dict[str, ActivationIntervention] | None
    ) = None,
) -> Tensor:
    """Collect CPU logits at each example's supervised answer marker."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split.samples == 0:
        raise ValueError("cannot evaluate an empty split")
    device = _model_device(model)
    was_training = model.training
    model.eval()
    rows: list[Tensor] = []
    try:
        for start in range(0, split.samples, batch_size):
            stop = min(start + batch_size, split.samples)
            output = model(
                split.input_ids[start:stop].to(device),
                attention_mask=split.attention_mask[start:stop].to(device),
                activation_interventions=activation_interventions,
            )
            positions = split.supervised_positions[start:stop].to(device)
            batch_rows = torch.arange(stop - start, device=device)
            rows.append(output.logits[batch_rows, positions].cpu())
    finally:
        model.train(was_training)
    return torch.cat(rows, dim=0)


def variable_associative_metrics_from_logits(
    split: VariableAssociativeRecallSplit,
    answer_logits: Tensor,
) -> VariableAssociativeMetrics:
    """Compute aggregate and every predeclared layout/length stratum."""

    if (
        not isinstance(answer_logits, Tensor)
        or answer_logits.ndim != 2
        or answer_logits.shape[0] != split.samples
    ):
        raise ValueError("answer_logits must have shape [split samples, vocab]")
    targets = split.answer_token_ids
    predictions = answer_logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    probabilities = answer_logits.softmax(dim=-1)
    correct_probabilities = probabilities.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    query_accuracies = tuple(
        float(correct[split.query_slots == value].float().mean().item())
        for value in range(2)
    )
    pair_order_accuracies = tuple(
        float(correct[split.pair_orders == value].float().mean().item())
        for value in range(2)
    )
    layout_accuracies = tuple(
        float(correct[split.layout_indices == value].float().mean().item())
        for value in range(int(split.layout_indices.max().item()) + 1)
    )
    lengths = tuple(
        int(value)
        for value in torch.unique(split.valid_lengths, sorted=True).tolist()
    )
    length_accuracies = tuple(
        (
            length,
            float(correct[split.valid_lengths == length].float().mean().item()),
        )
        for length in lengths
    )

    correct_per_context = torch.bincount(
        split.example_context_indices,
        weights=correct.to(torch.float32),
        minlength=split.contexts,
    )
    samples_per_context = torch.bincount(
        split.example_context_indices,
        minlength=split.contexts,
    )
    paired = correct_per_context.eq(samples_per_context)
    return VariableAssociativeMetrics(
        samples=split.samples,
        contexts=split.contexts,
        hard_nll=float(F.cross_entropy(answer_logits, targets).item()),
        answer_accuracy=float(correct.float().mean().item()),
        paired_context_accuracy=float(paired.float().mean().item()),
        query_accuracies=(query_accuracies[0], query_accuracies[1]),
        pair_order_accuracies=(
            pair_order_accuracies[0],
            pair_order_accuracies[1],
        ),
        layout_accuracies=layout_accuracies,
        length_accuracies=length_accuracies,
        minimum_query_accuracy=min(query_accuracies),
        minimum_pair_order_accuracy=min(pair_order_accuracies),
        minimum_layout_accuracy=min(layout_accuracies),
        minimum_length_accuracy=min(value for _, value in length_accuracies),
        mean_correct_probability=float(correct_probabilities.mean().item()),
    )


def evaluate_variable_associative_recall(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    batch_size: int = 512,
    activation_interventions: (
        dict[str, ActivationIntervention] | None
    ) = None,
) -> VariableAssociativeMetrics:
    return variable_associative_metrics_from_logits(
        split,
        variable_associative_answer_logits(
            model,
            split,
            batch_size=batch_size,
            activation_interventions=activation_interventions,
        ),
    )


def _snapshot(model: ToyTransformer) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _is_better(
    candidate: VariableAssociativeMetrics,
    current: VariableAssociativeMetrics | None,
) -> bool:
    if current is None:
        return True
    return (
        candidate.paired_context_accuracy,
        candidate.answer_accuracy,
        candidate.minimum_stratum_accuracy,
        -candidate.hard_nll,
    ) > (
        current.paired_context_accuracy,
        current.answer_accuracy,
        current.minimum_stratum_accuracy,
        -current.hard_nll,
    )


def _passes(
    metrics: VariableAssociativeMetrics,
    config: VariableAssociativeTrainingConfig,
) -> bool:
    return (
        metrics.answer_accuracy >= config.minimum_answer_accuracy
        and metrics.paired_context_accuracy
        >= config.minimum_paired_context_accuracy
        and metrics.minimum_stratum_accuracy >= config.minimum_stratum_accuracy
    )


def train_variable_associative_recall(
    model: ToyTransformer,
    splits: VariableAssociativeRecallSplits,
    config: VariableAssociativeTrainingConfig | None = None,
) -> VariableAssociativeTrainingResult:
    """Train, restore the best validation checkpoint, and evaluate test once."""

    options = config or VariableAssociativeTrainingConfig()
    device = torch.device(options.device)
    model.to(device)
    torch.use_deterministic_algorithms(options.deterministic_algorithms)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        betas=options.betas,
        eps=options.epsilon,
        weight_decay=options.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        options.shuffle_seed
    )
    permutation = torch.randperm(splits.train.samples, generator=generator)
    cursor = 0
    best: VariableAssociativeCheckpoint | None = None
    history: list[VariableAssociativeEvaluation] = []
    consecutive = 0
    converged = False
    final_step = 0

    for step in range(1, options.max_steps + 1):
        if cursor + options.batch_size > splits.train.samples:
            permutation = torch.randperm(
                splits.train.samples,
                generator=generator,
            )
            cursor = 0
        indices = permutation[cursor : cursor + options.batch_size]
        cursor += options.batch_size
        inputs = splits.train.input_ids.index_select(0, indices).to(device)
        attention_mask = splits.train.attention_mask.index_select(
            0,
            indices,
        ).to(device)
        targets = splits.train.targets.index_select(0, indices).to(device)
        model.train()
        output = model(inputs, attention_mask=attention_mask)
        loss = F.cross_entropy(
            output.logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=splits.task_config.ignore_index,
            label_smoothing=options.label_smoothing,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            options.gradient_clip_norm,
        )
        optimizer.step()
        final_step = step

        if (
            step % options.evaluation_interval != 0
            and step != options.max_steps
        ):
            continue
        train_metrics = evaluate_variable_associative_recall(
            model,
            splits.train,
            batch_size=options.evaluation_batch_size,
        )
        validation_metrics = evaluate_variable_associative_recall(
            model,
            splits.validation,
            batch_size=options.evaluation_batch_size,
        )
        history.append(
            VariableAssociativeEvaluation(
                step=step,
                batch_training_loss=float(loss.detach().item()),
                train=train_metrics,
                validation=validation_metrics,
            )
        )
        if _is_better(
            validation_metrics,
            None if best is None else best.validation_metrics,
        ):
            best = VariableAssociativeCheckpoint(
                step=step,
                model_state_dict=_snapshot(model),
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
        if _passes(validation_metrics, options):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= options.required_consecutive_evaluations:
            converged = True
            break

    if best is None:
        raise RuntimeError("training completed without an evaluation checkpoint")
    model.load_state_dict(best.model_state_dict)
    model.eval()
    test_metrics = evaluate_variable_associative_recall(
        model,
        splits.test,
        batch_size=options.evaluation_batch_size,
    )
    return VariableAssociativeTrainingResult(
        model=model,
        model_config=model.config,
        task_config=splits.task_config,
        training_config=options,
        splits=splits,
        best_checkpoint=best,
        history=tuple(history),
        final_step=final_step,
        converged=converged,
        test_metrics=test_metrics,
    )


def run_variable_associative_training(
    *,
    task_config: VariableAssociativeRecallTaskConfig | None = None,
    model_config: TransformerConfig | None = None,
    training_config: VariableAssociativeTrainingConfig | None = None,
) -> VariableAssociativeTrainingResult:
    """Construct the deterministic task/model and run training."""

    task = task_config or VariableAssociativeRecallTaskConfig()
    configured_model = model_config or variable_associative_recall_model_config(
        task
    )
    if configured_model.vocab_size != task.vocab_size:
        raise ValueError("model vocab_size must match the task")
    if configured_model.max_sequence_length < task.maximum_sequence_length:
        raise ValueError("model max_sequence_length is shorter than the task")
    options = training_config or VariableAssociativeTrainingConfig()
    torch.manual_seed(options.model_seed)
    torch.use_deterministic_algorithms(options.deterministic_algorithms)
    model = ToyTransformer(configured_model)
    splits = build_variable_associative_recall_splits(task)
    return train_variable_associative_recall(model, splits, options)


def _task_config_from_dict(
    raw: dict[str, object],
) -> VariableAssociativeRecallTaskConfig:
    values = copy.deepcopy(raw)
    layouts_raw = values.get("layouts")
    if not isinstance(layouts_raw, (list, tuple)):
        raise ValueError("checkpoint task layouts are invalid")
    values["layouts"] = tuple(
        VariableAssociativeLayout(**layout)
        for layout in layouts_raw
        if isinstance(layout, dict)
    )
    if len(values["layouts"]) != len(layouts_raw):  # type: ignore[arg-type]
        raise ValueError("checkpoint task layout entry is invalid")
    return VariableAssociativeRecallTaskConfig(**values)  # type: ignore[arg-type]


def save_variable_associative_checkpoint(
    result: VariableAssociativeTrainingResult,
    path: str | Path,
) -> Path:
    """Save a reproducible local source checkpoint for compiler experiments."""

    if not isinstance(result, VariableAssociativeTrainingResult):
        raise TypeError("result must be a VariableAssociativeTrainingResult")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "format_version": _CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "task_config": asdict(result.task_config),
        "training_config": asdict(result.training_config),
        "dataset_sha256": result.splits.dataset_sha256,
        "split_sha256": {
            "train": result.splits.train.content_sha256,
            "validation": result.splits.validation.content_sha256,
            "test": result.splits.test.content_sha256,
        },
        "best_step": result.best_checkpoint.step,
        "final_step": result.final_step,
        "converged": result.converged,
        "train_metrics": asdict(result.best_checkpoint.train_metrics),
        "validation_metrics": asdict(
            result.best_checkpoint.validation_metrics
        ),
        "test_metrics": asdict(result.test_metrics),
        "model_state_dict": _snapshot(result.model),
    }
    torch.save(payload, destination)
    return destination


def load_variable_associative_checkpoint(
    path: str | Path,
) -> tuple[ToyTransformer, VariableAssociativeRecallSplits, dict[str, object]]:
    """Strict-load a local checkpoint and reproduce its data split hashes."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    expected = {
        "schema",
        "format_version",
        "model_config",
        "task_config",
        "training_config",
        "dataset_sha256",
        "split_sha256",
        "best_step",
        "final_step",
        "converged",
        "train_metrics",
        "validation_metrics",
        "test_metrics",
        "model_state_dict",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("variable associative checkpoint fields are invalid")
    if (
        raw["schema"] != _CHECKPOINT_SCHEMA
        or raw["format_version"] != _CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported variable associative checkpoint")
    if not isinstance(raw["model_config"], dict):
        raise ValueError("checkpoint model_config is invalid")
    if not isinstance(raw["task_config"], dict):
        raise ValueError("checkpoint task_config is invalid")
    task = _task_config_from_dict(raw["task_config"])
    model_config = TransformerConfig(**raw["model_config"])
    splits = build_variable_associative_recall_splits(task)
    expected_split_hashes = {
        "train": splits.train.content_sha256,
        "validation": splits.validation.content_sha256,
        "test": splits.test.content_sha256,
    }
    if (
        raw["dataset_sha256"] != splits.dataset_sha256
        or raw["split_sha256"] != expected_split_hashes
    ):
        raise ValueError("checkpoint dataset binding is invalid")
    if not isinstance(raw["model_state_dict"], dict):
        raise ValueError("checkpoint model_state_dict is invalid")
    model = ToyTransformer(model_config)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key != "model_state_dict"
    }
    return model, splits, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the variable-layout associative-recall source model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_variable_associative_training(
        training_config=VariableAssociativeTrainingConfig(
            device=arguments.device,
            max_steps=arguments.max_steps,
            evaluation_interval=arguments.evaluation_interval,
            batch_size=arguments.batch_size,
            evaluation_batch_size=arguments.evaluation_batch_size,
        )
    )
    output = save_variable_associative_checkpoint(result, arguments.output)
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "converged": result.converged,
                "final_step": result.final_step,
                "best_step": result.best_checkpoint.step,
                "dataset_sha256": result.splits.dataset_sha256,
                "train": asdict(result.best_checkpoint.train_metrics),
                "validation": asdict(
                    result.best_checkpoint.validation_metrics
                ),
                "test": asdict(result.test_metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT",
    "VariableAssociativeCheckpoint",
    "VariableAssociativeEvaluation",
    "VariableAssociativeMetrics",
    "VariableAssociativeTrainingConfig",
    "VariableAssociativeTrainingResult",
    "evaluate_variable_associative_recall",
    "load_variable_associative_checkpoint",
    "run_variable_associative_training",
    "save_variable_associative_checkpoint",
    "train_variable_associative_recall",
    "variable_associative_answer_logits",
    "variable_associative_metrics_from_logits",
]


if __name__ == "__main__":
    raise SystemExit(main())
