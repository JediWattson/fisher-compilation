"""Deterministic toy tasks for training and decomposition experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModularAdditionTask:
    """All ``a + b mod p`` equations as a final-token prediction task."""

    modulus: int = 13
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.modulus < 2:
            raise ValueError("modulus must be at least 2")

    @property
    def bos_token(self) -> int:
        return self.modulus

    @property
    def plus_token(self) -> int:
        return self.modulus + 1

    @property
    def equals_token(self) -> int:
        return self.modulus + 2

    @property
    def vocab_size(self) -> int:
        return self.modulus + 3

    @property
    def sequence_length(self) -> int:
        return 5

    def dataset(self) -> tuple[Tensor, Tensor]:
        """Return every equation and targets supervised only at ``=``."""

        equations: list[list[int]] = []
        answers: list[int] = []
        for left in range(self.modulus):
            for right in range(self.modulus):
                equations.append(
                    [
                        self.bos_token,
                        left,
                        self.plus_token,
                        right,
                        self.equals_token,
                    ]
                )
                answers.append((left + right) % self.modulus)
        inputs = torch.tensor(equations, dtype=torch.long)
        targets = torch.full_like(inputs, self.ignore_index)
        targets[:, -1] = torch.tensor(answers, dtype=torch.long)
        return inputs, targets

    def decode_equation(self, input_ids: Tensor, prediction: int) -> str:
        left = input_ids[1].item()
        right = input_ids[3].item()
        return f"{left} + {right} mod {self.modulus} = {prediction}"

