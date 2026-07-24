"""Transformer components with explicit instrumentation points."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .config import TransformerConfig


class LayerExecutor(nn.Module, ABC):
    """Stable boundary between the model and one layer implementation."""

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        """Execute one layer and return hidden states with the same shape."""


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        batch_size, sequence_length, d_model = hidden_states.shape

        qkv = self.qkv(hidden_states)
        qkv = qkv.view(
            batch_size, sequence_length, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = record(trace, f"{prefix}.query", query.transpose(1, 2))
        key = record(trace, f"{prefix}.key", key.transpose(1, 2))
        value = record(trace, f"{prefix}.value", value.transpose(1, 2))

        scores = record(
            trace, f"{prefix}.scores", torch.matmul(query, key.transpose(-2, -1)) * self.scale
        )
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden_states.device,
        ).tril()
        allowed = causal_mask.view(1, 1, sequence_length, sequence_length)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].to(torch.bool)
        masked_scores = record(
            trace,
            f"{prefix}.masked_scores",
            scores.masked_fill(~allowed, torch.finfo(scores.dtype).min),
        )
        probabilities = record(
            trace, f"{prefix}.probabilities", torch.softmax(masked_scores, dim=-1)
        )
        dropped_probabilities = self.attention_dropout(probabilities)

        context_heads = record(
            trace,
            f"{prefix}.context_heads",
            torch.matmul(dropped_probabilities, value),
        )
        context = context_heads.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        output = self.output_dropout(self.output_projection(context))
        return record(trace, f"{prefix}.output", output)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.d_model, config.d_ff)
        self.output_projection = nn.Linear(config.d_ff, config.d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        pre_activation = record(
            trace, f"{prefix}.pre_activation", self.input_projection(hidden_states)
        )
        activated = record(
            trace, f"{prefix}.activated", self.activation(pre_activation)
        )
        output = self.dropout(self.output_projection(activated))
        return record(trace, f"{prefix}.output", output)


class TransformerBlock(LayerExecutor):
    """A small pre-norm transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        normalized = record(trace, f"{prefix}.ln1", self.norm1(hidden_states))
        attention = self.attention(
            normalized,
            attention_mask=attention_mask,
            trace=trace,
            prefix=f"{prefix}.attention",
        )
        post_attention = record(
            trace, f"{prefix}.post_attention", hidden_states + attention
        )
        normalized = record(trace, f"{prefix}.ln2", self.norm2(post_attention))
        mlp_output = self.mlp(
            normalized, trace=trace, prefix=f"{prefix}.mlp"
        )
        return record(trace, f"{prefix}.output", post_attention + mlp_output)

