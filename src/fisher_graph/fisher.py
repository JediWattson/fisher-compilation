"""Empirical diagonal Fisher information over captured activations."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import ToyTransformer

ActivationSelector = Collection[str] | Callable[[str], bool] | None


@dataclass(frozen=True, slots=True)
class FisherEntry:
    """Per-coordinate empirical diagonal Fisher for one activation."""

    diagonal: Tensor
    samples: int

    @property
    def mean(self) -> Tensor:
        return self.diagonal.mean()

    @property
    def total(self) -> Tensor:
        return self.diagonal.sum()


@dataclass(frozen=True, slots=True)
class FisherReport:
    activations: Mapping[str, FisherEntry]
    mean_loss: float
    samples: int

    def ranked(self) -> list[tuple[str, float]]:
        """Activation names ranked by mean diagonal Fisher."""

        return sorted(
            (
                (name, entry.mean.item())
                for name, entry in self.activations.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )


def _is_selected(name: str, selector: ActivationSelector) -> bool:
    if selector is None:
        return True
    if callable(selector):
        return selector(name)
    return name in selector


def empirical_activation_fisher(
    model: ToyTransformer,
    input_ids: Tensor,
    targets: Tensor,
    *,
    attention_mask: Tensor | None = None,
    activations: ActivationSelector = None,
    ignore_index: int = -100,
) -> FisherReport:
    """Estimate empirical diagonal Fisher over activations.

    For each example this computes the gradient of mean token negative
    log-likelihood with respect to each selected activation and accumulates its
    elementwise square. Per-example gradients are essential here: squaring one
    batch-averaged gradient would include cross-example terms and would not be
    the empirical Fisher.

    The leading batch dimension is removed from every returned diagonal, so an
    activation shaped ``[batch, sequence, width]`` becomes
    ``[sequence, width]`` in the report. All examples must share a sequence
    length.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if targets.shape != input_ids.shape:
        raise ValueError("targets must have the same shape as input_ids")
    if input_ids.shape[0] == 0:
        raise ValueError("cannot estimate Fisher from an empty batch")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    was_training = model.training
    model.eval()
    accumulated: dict[str, Tensor] = {}
    loss_total = 0.0
    sample_count = input_ids.shape[0]

    try:
        for sample_index in range(sample_count):
            sample_inputs = input_ids[sample_index : sample_index + 1]
            sample_targets = targets[sample_index : sample_index + 1]
            sample_mask = (
                attention_mask[sample_index : sample_index + 1]
                if attention_mask is not None
                else None
            )
            if not (sample_targets != ignore_index).any():
                raise ValueError(
                    f"sample {sample_index} has no targets outside ignore_index"
                )

            output = model(
                sample_inputs,
                attention_mask=sample_mask,
                capture_activations=True,
                retain_activation_gradients=False,
            )
            assert output.activations is not None
            selected = [
                (name, tensor)
                for name, tensor in output.activations.items()
                if tensor.requires_grad and _is_selected(name, activations)
            ]
            if not selected:
                raise ValueError("the activation selector matched no differentiable taps")

            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                sample_targets.reshape(-1),
                ignore_index=ignore_index,
                reduction="mean",
            )
            gradients = torch.autograd.grad(
                loss,
                tuple(tensor for _, tensor in selected),
                allow_unused=True,
            )
            loss_total += loss.detach().item()

            for (name, tensor), gradient in zip(selected, gradients, strict=True):
                if gradient is None:
                    diagonal = torch.zeros_like(tensor[0], memory_format=torch.preserve_format)
                else:
                    diagonal = gradient.detach()[0].square()
                if name not in accumulated:
                    accumulated[name] = diagonal
                else:
                    accumulated[name] = accumulated[name] + diagonal
    finally:
        model.train(was_training)

    entries = {
        name: FisherEntry(
            diagonal=(diagonal / sample_count).cpu(),
            samples=sample_count,
        )
        for name, diagonal in accumulated.items()
    }
    return FisherReport(
        activations=entries,
        mean_loss=loss_total / sample_count,
        samples=sample_count,
    )

