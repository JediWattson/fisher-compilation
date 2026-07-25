"""Model-independent calibration batches and score objectives.

The adapter owns model execution; the calibration batch owns task data; and
the score objective owns the scalar whose activation gradient defines the
empirical Fisher sample. Keeping those responsibilities separate lets the
same compiler analyze associative recall, ordinary causal language modeling,
or another objective without teaching a model adapter about the task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from ..adapters import AdapterRun


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    """One possibly padded batch from a replayable calibration stream.

    Tensor inputs use batch axis zero unless their names appear in
    ``shared_input_names``. This makes unbatched values such as
    ``cache_position: [query]`` unambiguous even when query length happens to
    equal batch size. ``example_ids`` bind replayed sequence contexts to
    captured activation rows for context-sensitive Jacobian analysis.
    """

    model_inputs: Mapping[str, Tensor]
    targets: Tensor
    valid_positions: Tensor
    shared_input_names: frozenset[str] = field(default_factory=frozenset)
    example_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_inputs, Mapping) or not self.model_inputs:
            raise ValueError("model_inputs must be a nonempty mapping")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, Tensor)
            for name, value in self.model_inputs.items()
        ):
            raise TypeError(
                "model_inputs must map nonempty strings to Tensors"
            )
        if not isinstance(self.shared_input_names, frozenset):
            raise TypeError("shared_input_names must be a frozenset")
        if not self.shared_input_names.issubset(self.model_inputs):
            raise ValueError(
                "shared_input_names must reference declared model inputs"
            )
        if not isinstance(self.targets, Tensor):
            raise TypeError("targets must be a Tensor")
        if not isinstance(self.valid_positions, Tensor):
            raise TypeError("valid_positions must be a Tensor")
        if self.targets.ndim < 1:
            raise ValueError("targets must include a batch dimension")
        if self.valid_positions.ndim != 2:
            raise ValueError(
                "valid_positions must have shape [batch, sequence]"
            )
        if self.valid_positions.dtype is not torch.bool:
            raise ValueError("valid_positions must be boolean")
        if self.targets.shape[0] != self.valid_positions.shape[0]:
            raise ValueError(
                "targets and valid_positions must share a batch size"
            )
        batch_size = self.targets.shape[0]
        for name, value in self.model_inputs.items():
            if (
                value.ndim > 0
                and name not in self.shared_input_names
                and value.shape[0] != batch_size
            ):
                raise ValueError(
                    f"model input {name!r} must have batch size "
                    f"{batch_size} or be declared shared"
                )
        if self.example_ids is not None:
            if (
                type(self.example_ids) is not tuple
                or len(self.example_ids) != batch_size
                or any(
                    not isinstance(example_id, str) or not example_id
                    for example_id in self.example_ids
                )
            ):
                raise ValueError(
                    "example_ids must contain one nonempty string per example"
                )
            if len(set(self.example_ids)) != len(self.example_ids):
                raise ValueError("example_ids must be unique within a batch")

    @property
    def batch_size(self) -> int:
        return self.targets.shape[0]

    def sample(self, index: int) -> CalibrationBatch:
        """Return one example while preserving adapter-owned shared tensors."""

        if type(index) is not int or not 0 <= index < self.batch_size:
            raise IndexError("calibration sample index is out of range")
        inputs: dict[str, Tensor] = {}
        for name, value in self.model_inputs.items():
            if value.ndim > 0 and name not in self.shared_input_names:
                inputs[name] = value[index : index + 1]
            else:
                inputs[name] = value
        return CalibrationBatch(
            model_inputs=inputs,
            targets=self.targets[index : index + 1],
            valid_positions=self.valid_positions[index : index + 1],
            shared_input_names=self.shared_input_names,
            example_ids=(
                None
                if self.example_ids is None
                else (self.example_ids[index],)
            ),
        )


class ScoreObjective(Protocol):
    """Return one differentiable scalar score for a single-example run."""

    def __call__(
        self,
        run: AdapterRun,
        batch: CalibrationBatch,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class CausalLanguageModelNLL:
    """Summed hard-target negative log likelihood for a decoder batch."""

    ignore_index: int = -100

    def __post_init__(self) -> None:
        if type(self.ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")

    def __call__(
        self,
        run: AdapterRun,
        batch: CalibrationBatch,
    ) -> Tensor:
        if run.logits.ndim != 3:
            raise ValueError(
                "causal language-model logits must have shape "
                "[batch, sequence, vocabulary]"
            )
        if batch.targets.shape != run.logits.shape[:2]:
            raise ValueError(
                "causal language-model targets must match logits positions"
            )
        targets = batch.targets.to(device=run.logits.device)
        valid_positions = batch.valid_positions.to(device=run.logits.device)
        supervised = targets != self.ignore_index
        if (supervised & ~valid_positions).any():
            raise ValueError(
                "supervised targets must be at valid calibration positions"
            )
        if not supervised.any():
            raise ValueError("a calibration sample has no supervised target")
        logits = run.logits
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="sum",
        )


__all__ = [
    "CalibrationBatch",
    "CausalLanguageModelNLL",
    "ScoreObjective",
]
