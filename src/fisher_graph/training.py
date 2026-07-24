"""Training utilities for the modular-addition experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TransformerConfig
from .model import ToyTransformer
from .task import ModularAdditionTask


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 17
    steps: int = 2_500
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    target_accuracy: float = 1.0
    target_loss: float = 0.02
    minimum_steps: int = 100
    log_every: int = 100


@dataclass(frozen=True, slots=True)
class TrainingResult:
    steps: int
    loss: float
    accuracy: float
    history: tuple[dict[str, float | int], ...]


def task_loss(logits: Tensor, targets: Tensor, *, ignore_index: int) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def evaluate_task(
    model: ToyTransformer,
    inputs: Tensor,
    targets: Tensor,
    *,
    ignore_index: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    output = model(inputs)
    loss = task_loss(output.logits, targets, ignore_index=ignore_index)
    supervised = targets != ignore_index
    predictions = output.logits.argmax(dim=-1)
    accuracy = (predictions[supervised] == targets[supervised]).float().mean()
    model.train(was_training)
    return loss.item(), accuracy.item()


def train_modular_addition(
    model: ToyTransformer,
    task: ModularAdditionTask,
    config: TrainingConfig,
    *,
    verbose: bool = True,
) -> TrainingResult:
    """Train deterministically on the complete modular-addition table."""

    torch.manual_seed(config.seed)
    inputs, targets = task.dataset()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    final_loss = float("inf")
    final_accuracy = 0.0

    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = task_loss(
            output.logits,
            targets,
            ignore_index=task.ignore_index,
        )
        loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % config.log_every == 0
            or step == config.steps
        )
        if should_evaluate:
            final_loss, final_accuracy = evaluate_task(
                model,
                inputs,
                targets,
                ignore_index=task.ignore_index,
            )
            point: dict[str, float | int] = {
                "step": step,
                "loss": final_loss,
                "accuracy": final_accuracy,
            }
            history.append(point)
            if verbose:
                print(
                    f"step={step:4d} loss={final_loss:.6f} "
                    f"accuracy={final_accuracy:.3%}",
                    flush=True,
                )
            if (
                step >= config.minimum_steps
                and final_accuracy >= config.target_accuracy
                and final_loss <= config.target_loss
            ):
                break

    return TrainingResult(
        steps=step,
        loss=final_loss,
        accuracy=final_accuracy,
        history=tuple(history),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: ToyTransformer,
    task: ModularAdditionTask,
    training_config: TrainingConfig,
    result: TrainingResult,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "task": {
                "name": "modular_addition",
                "modulus": task.modulus,
                "ignore_index": task.ignore_index,
            },
            "training_config": asdict(training_config),
            "training_result": {
                "steps": result.steps,
                "loss": result.loss,
                "accuracy": result.accuracy,
                "history": list(result.history),
            },
        },
        Path(path),
    )


def load_checkpoint(path: str | Path) -> tuple[ToyTransformer, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = ToyTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

