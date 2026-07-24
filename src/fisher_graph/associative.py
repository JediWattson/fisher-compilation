"""Deterministic two-pair associative-recall training task.

The task presents two key/value pairs followed by a query key::

    BOS k0 v0 k1 v1 QUERY kq ANSWER

Only the final position is supervised, where the model must predict ``vq``.
Keys and values use disjoint token ranges, and both keys and both values are
distinct.  Contexts are split as groups, so the two query variants for a
context always remain in the same train, validation, or test partition.

The defaults are intentionally small enough for a quick deterministic CPU
run.  Label smoothing prevents a correct model from becoming so saturated
that observed-label activation-Fisher gradients collapse toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import torch
import torch.nn.functional as F
from torch import Tensor

from .activations import ActivationIntervention
from .config import TransformerConfig
from .model import ToyTransformer


@dataclass(frozen=True, slots=True)
class AssociativeRecallTaskConfig:
    """Token vocabulary and grouped data-split configuration."""

    n_keys: int = 8
    n_values: int = 8
    split_seed: int = 1729
    train_fraction: float = 0.8
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.n_keys < 2:
            raise ValueError("n_keys must be at least 2")
        if self.n_values < 2:
            raise ValueError("n_values must be at least 2")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        if 0 <= self.ignore_index < self.vocab_size:
            raise ValueError("ignore_index must not be a task token ID")

    @property
    def value_offset(self) -> int:
        return self.n_keys

    @property
    def bos_token_id(self) -> int:
        return self.n_keys + self.n_values

    @property
    def query_token_id(self) -> int:
        return self.bos_token_id + 1

    @property
    def answer_token_id(self) -> int:
        return self.bos_token_id + 2

    @property
    def vocab_size(self) -> int:
        return self.n_keys + self.n_values + 3

    @property
    def sequence_length(self) -> int:
        return 8

    @property
    def context_count(self) -> int:
        return (
            self.n_keys
            * (self.n_keys - 1)
            * self.n_values
            * (self.n_values - 1)
        )


@dataclass(frozen=True, slots=True)
class AssociativeRecallTrainingConfig:
    """Optimizer, reproducibility, evaluation, and stopping settings."""

    model_seed: int = 1729
    shuffle_seed: int = 1730
    learning_rate: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    batch_size: int = 128
    evaluation_batch_size: int = 256
    max_steps: int = 3000
    evaluation_interval: int = 100
    gradient_clip_norm: float = 1.0
    minimum_answer_accuracy: float = 0.995
    minimum_paired_accuracy: float = 0.99
    minimum_query_accuracy: float = 0.99
    required_consecutive_evaluations: int = 2
    deterministic_algorithms: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        beta1, beta2 = self.betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_steps <= 0 or self.evaluation_interval <= 0:
            raise ValueError("step counts must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        for name, value in (
            ("minimum_answer_accuracy", self.minimum_answer_accuracy),
            ("minimum_paired_accuracy", self.minimum_paired_accuracy),
            ("minimum_query_accuracy", self.minimum_query_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.required_consecutive_evaluations <= 0:
            raise ValueError("required_consecutive_evaluations must be positive")


@dataclass(frozen=True, slots=True)
class AssociativeRecallSplit:
    """One split, with context membership retained for paired metrics."""

    name: str
    input_ids: Tensor
    targets: Tensor
    query_slots: Tensor
    answer_value_indices: Tensor
    example_context_indices: Tensor
    context_ids: Tensor
    n_values: int

    @property
    def samples(self) -> int:
        return self.input_ids.shape[0]

    @property
    def contexts(self) -> int:
        return self.context_ids.shape[0]


@dataclass(frozen=True, slots=True)
class AssociativeRecallSplits:
    """Grouped train, Fisher/validation, and untouched test partitions."""

    task_config: AssociativeRecallTaskConfig
    train: AssociativeRecallSplit
    validation: AssociativeRecallSplit
    test: AssociativeRecallSplit


@dataclass(frozen=True, slots=True)
class AssociativeRecallMetrics:
    """Hard-target task metrics for one split."""

    samples: int
    contexts: int
    hard_nll: float
    answer_accuracy: float
    paired_context_accuracy: float
    query_accuracies: tuple[float, float]
    minimum_query_accuracy: float
    value_accuracies: tuple[float, ...]
    minimum_value_accuracy: float
    mean_correct_probability: float


@dataclass(frozen=True, slots=True)
class AssociativeRecallEvaluation:
    """Metrics captured at a training evaluation boundary."""

    step: int
    batch_training_loss: float
    train: AssociativeRecallMetrics
    validation: AssociativeRecallMetrics


@dataclass(frozen=True, slots=True)
class AssociativeRecallCheckpoint:
    """Best validation checkpoint, detached and copied onto CPU."""

    step: int
    model_state_dict: dict[str, Tensor]
    train_metrics: AssociativeRecallMetrics
    validation_metrics: AssociativeRecallMetrics


@dataclass(slots=True)
class AssociativeRecallTrainingResult:
    """A trained model plus all state needed by an experiment runner."""

    model: ToyTransformer
    model_config: TransformerConfig
    task_config: AssociativeRecallTaskConfig
    training_config: AssociativeRecallTrainingConfig
    splits: AssociativeRecallSplits
    best_checkpoint: AssociativeRecallCheckpoint
    history: tuple[AssociativeRecallEvaluation, ...]
    final_step: int
    converged: bool
    test_metrics: AssociativeRecallMetrics


def associative_recall_model_config(
    task_config: AssociativeRecallTaskConfig | None = None,
) -> TransformerConfig:
    """Return the default instrumentable model configuration for the task."""

    task = task_config or AssociativeRecallTaskConfig()
    return TransformerConfig(
        vocab_size=task.vocab_size,
        max_sequence_length=task.sequence_length,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        tie_embeddings=False,
    )


def _enumerate_contexts(config: AssociativeRecallTaskConfig) -> Tensor:
    contexts = [
        (key0, key1, value0, value1)
        for key0, key1 in permutations(range(config.n_keys), 2)
        for value0, value1 in permutations(range(config.n_values), 2)
    ]
    return torch.tensor(contexts, dtype=torch.long)


def _make_split(
    name: str,
    contexts: Tensor,
    context_ids: Tensor,
    config: AssociativeRecallTaskConfig,
) -> AssociativeRecallSplit:
    selected = contexts.index_select(0, context_ids)
    context_count = selected.shape[0]
    sample_count = 2 * context_count

    input_ids = torch.empty(
        (sample_count, config.sequence_length), dtype=torch.long
    )
    targets = torch.full_like(input_ids, config.ignore_index)
    query_slots = torch.arange(2, dtype=torch.long).repeat(context_count)
    example_context_indices = torch.arange(
        context_count, dtype=torch.long
    ).repeat_interleave(2)

    keys = selected[:, :2].repeat_interleave(2, dim=0)
    values = selected[:, 2:].repeat_interleave(2, dim=0)
    rows = torch.arange(sample_count)
    queried_keys = keys[rows, query_slots]
    answer_value_indices = values[rows, query_slots]

    input_ids[:, 0] = config.bos_token_id
    input_ids[:, 1] = keys[:, 0]
    input_ids[:, 2] = config.value_offset + values[:, 0]
    input_ids[:, 3] = keys[:, 1]
    input_ids[:, 4] = config.value_offset + values[:, 1]
    input_ids[:, 5] = config.query_token_id
    input_ids[:, 6] = queried_keys
    input_ids[:, 7] = config.answer_token_id
    targets[:, -1] = config.value_offset + answer_value_indices

    return AssociativeRecallSplit(
        name=name,
        input_ids=input_ids,
        targets=targets,
        query_slots=query_slots,
        answer_value_indices=answer_value_indices,
        example_context_indices=example_context_indices,
        context_ids=context_ids.clone(),
        n_values=config.n_values,
    )


def build_associative_recall_splits(
    config: AssociativeRecallTaskConfig | None = None,
) -> AssociativeRecallSplits:
    """Enumerate and deterministically split complete two-query contexts.

    The configured training fraction is floored to a whole context.  The
    remaining contexts are divided equally between Fisher/validation and test,
    with test receiving one extra context if the remainder is odd.
    """

    task = config or AssociativeRecallTaskConfig()
    contexts = _enumerate_contexts(task)
    generator = torch.Generator(device="cpu").manual_seed(task.split_seed)
    permutation = torch.randperm(contexts.shape[0], generator=generator)

    train_contexts = int(contexts.shape[0] * task.train_fraction)
    remaining = contexts.shape[0] - train_contexts
    validation_contexts = remaining // 2
    if train_contexts == 0 or validation_contexts == 0:
        raise ValueError("the configured task is too small for three nonempty splits")

    train_end = train_contexts
    validation_end = train_end + validation_contexts
    train_ids = permutation[:train_end]
    validation_ids = permutation[train_end:validation_end]
    test_ids = permutation[validation_end:]

    return AssociativeRecallSplits(
        task_config=task,
        train=_make_split("train", contexts, train_ids, task),
        validation=_make_split(
            "validation", contexts, validation_ids, task
        ),
        test=_make_split("test", contexts, test_ids, task),
    )


def evaluate_associative_recall(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    activation_interventions: (
        dict[str, ActivationIntervention] | None
    ) = None,
) -> AssociativeRecallMetrics:
    """Evaluate hard-target accuracy, pairing, balance, and confidence."""

    answer_logits = associative_recall_answer_logits(
        model,
        split,
        batch_size=batch_size,
        device=device,
        activation_interventions=activation_interventions,
    )
    return associative_recall_metrics_from_logits(split, answer_logits)


def associative_recall_answer_logits(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    activation_interventions: (
        dict[str, ActivationIntervention] | None
    ) = None,
) -> Tensor:
    """Return CPU answer-position logits for one split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split.samples == 0:
        raise ValueError("cannot evaluate an empty split")

    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    if reference is None:
        raise ValueError(
            "model must contain at least one parameter or buffer"
        )
    model_device = reference.device
    evaluation_device = (
        model_device if device is None else torch.device(device)
    )
    if evaluation_device != model_device:
        raise ValueError(
            f"model is on {model_device}, but evaluation device is {evaluation_device}"
        )

    was_training = model.training
    model.eval()
    logits: list[Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, split.samples, batch_size):
                inputs = split.input_ids[start : start + batch_size].to(
                    evaluation_device
                )
                logits.append(
                    model(
                        inputs,
                        activation_interventions=activation_interventions,
                    ).logits[:, -1].cpu()
                )
    finally:
        model.train(was_training)

    return torch.cat(logits)


def associative_recall_metrics_from_logits(
    split: AssociativeRecallSplit,
    answer_logits: Tensor,
) -> AssociativeRecallMetrics:
    """Compute task metrics from answer-position logits."""

    if answer_logits.ndim != 2 or answer_logits.shape[0] != split.samples:
        raise ValueError("answer_logits must have shape [split samples, vocab]")
    answer_tokens = split.targets[:, -1]
    predictions = answer_logits.argmax(dim=-1)
    correct = predictions.eq(answer_tokens)
    probabilities = answer_logits.softmax(dim=-1)
    correct_probabilities = probabilities.gather(
        1, answer_tokens.unsqueeze(1)
    ).squeeze(1)

    query_accuracies = tuple(
        correct[split.query_slots == query_slot].float().mean().item()
        for query_slot in range(2)
    )
    value_accuracies = tuple(
        correct[split.answer_value_indices == value_index].float().mean().item()
        for value_index in range(split.n_values)
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
    paired_correct = correct_per_context.eq(samples_per_context)

    return AssociativeRecallMetrics(
        samples=split.samples,
        contexts=split.contexts,
        hard_nll=F.cross_entropy(answer_logits, answer_tokens).item(),
        answer_accuracy=correct.float().mean().item(),
        paired_context_accuracy=paired_correct.float().mean().item(),
        query_accuracies=(query_accuracies[0], query_accuracies[1]),
        minimum_query_accuracy=min(query_accuracies),
        value_accuracies=value_accuracies,
        minimum_value_accuracy=min(value_accuracies),
        mean_correct_probability=correct_probabilities.mean().item(),
    )


def _snapshot_state_dict(model: ToyTransformer) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _is_better(
    candidate: AssociativeRecallMetrics,
    current: AssociativeRecallMetrics | None,
) -> bool:
    if current is None:
        return True
    candidate_score = (
        candidate.paired_context_accuracy,
        candidate.answer_accuracy,
        candidate.minimum_query_accuracy,
        -candidate.hard_nll,
    )
    current_score = (
        current.paired_context_accuracy,
        current.answer_accuracy,
        current.minimum_query_accuracy,
        -current.hard_nll,
    )
    return candidate_score > current_score


def _meets_stopping_criteria(
    metrics: AssociativeRecallMetrics,
    config: AssociativeRecallTrainingConfig,
) -> bool:
    return (
        metrics.answer_accuracy >= config.minimum_answer_accuracy
        and metrics.paired_context_accuracy >= config.minimum_paired_accuracy
        and metrics.minimum_query_accuracy >= config.minimum_query_accuracy
    )


def train_associative_recall(
    model: ToyTransformer,
    splits: AssociativeRecallSplits,
    config: AssociativeRecallTrainingConfig | None = None,
) -> AssociativeRecallTrainingResult:
    """Train ``model``, restore its best validation state, and test it once."""

    training = config or AssociativeRecallTrainingConfig()
    device = torch.device(training.device)
    model.to(device)
    torch.use_deterministic_algorithms(training.deterministic_algorithms)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=training.betas,
        eps=training.epsilon,
        weight_decay=training.weight_decay,
    )
    shuffle_generator = torch.Generator(device="cpu").manual_seed(
        training.shuffle_seed
    )
    permutation = torch.randperm(
        splits.train.samples, generator=shuffle_generator
    )
    cursor = 0

    history: list[AssociativeRecallEvaluation] = []
    best_checkpoint: AssociativeRecallCheckpoint | None = None
    consecutive_successes = 0
    converged = False
    final_step = 0

    for step in range(1, training.max_steps + 1):
        if cursor + training.batch_size > splits.train.samples:
            permutation = torch.randperm(
                splits.train.samples, generator=shuffle_generator
            )
            cursor = 0
        indices = permutation[cursor : cursor + training.batch_size]
        cursor += training.batch_size

        inputs = splits.train.input_ids.index_select(0, indices).to(device)
        targets = splits.train.targets.index_select(0, indices).to(device)
        model.train()
        output = model(inputs)
        batch_loss = F.cross_entropy(
            output.logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=splits.task_config.ignore_index,
            label_smoothing=training.label_smoothing,
        )
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training.gradient_clip_norm
        )
        optimizer.step()
        final_step = step

        should_evaluate = (
            step % training.evaluation_interval == 0
            or step == training.max_steps
        )
        if not should_evaluate:
            continue

        train_metrics = evaluate_associative_recall(
            model,
            splits.train,
            batch_size=training.evaluation_batch_size,
            device=device,
        )
        validation_metrics = evaluate_associative_recall(
            model,
            splits.validation,
            batch_size=training.evaluation_batch_size,
            device=device,
        )
        evaluation = AssociativeRecallEvaluation(
            step=step,
            batch_training_loss=batch_loss.detach().item(),
            train=train_metrics,
            validation=validation_metrics,
        )
        history.append(evaluation)

        current_best = (
            best_checkpoint.validation_metrics
            if best_checkpoint is not None
            else None
        )
        if _is_better(validation_metrics, current_best):
            best_checkpoint = AssociativeRecallCheckpoint(
                step=step,
                model_state_dict=_snapshot_state_dict(model),
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )

        if _meets_stopping_criteria(validation_metrics, training):
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        if consecutive_successes >= training.required_consecutive_evaluations:
            converged = True
            break

    if best_checkpoint is None:
        raise RuntimeError("training completed without producing a checkpoint")

    model.load_state_dict(best_checkpoint.model_state_dict)
    model.eval()
    test_metrics = evaluate_associative_recall(
        model,
        splits.test,
        batch_size=training.evaluation_batch_size,
        device=device,
    )
    model.eval()

    return AssociativeRecallTrainingResult(
        model=model,
        model_config=model.config,
        task_config=splits.task_config,
        training_config=training,
        splits=splits,
        best_checkpoint=best_checkpoint,
        history=tuple(history),
        final_step=final_step,
        converged=converged,
        test_metrics=test_metrics,
    )


def run_associative_recall_experiment(
    *,
    task_config: AssociativeRecallTaskConfig | None = None,
    training_config: AssociativeRecallTrainingConfig | None = None,
    model_config: TransformerConfig | None = None,
) -> AssociativeRecallTrainingResult:
    """Construct the deterministic task and model, then run training."""

    task = task_config or AssociativeRecallTaskConfig()
    training = training_config or AssociativeRecallTrainingConfig()
    configured_model = model_config or associative_recall_model_config(task)
    if configured_model.vocab_size != task.vocab_size:
        raise ValueError(
            "model vocab_size must match the associative-recall vocabulary"
        )
    if configured_model.max_sequence_length < task.sequence_length:
        raise ValueError(
            "model max_sequence_length is shorter than the task sequence"
        )

    torch.manual_seed(training.model_seed)
    torch.use_deterministic_algorithms(training.deterministic_algorithms)
    model = ToyTransformer(configured_model)
    splits = build_associative_recall_splits(task)
    return train_associative_recall(model, splits, training)
