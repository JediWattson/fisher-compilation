"""The instrumented toy language model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .activations import ActivationIntervention, ActivationTrace, record
from .config import TransformerConfig
from .layers import LayerExecutor, TransformerBlock


@dataclass(slots=True)
class TransformerOutput:
    logits: Tensor
    activations: ActivationTrace | None


class ToyTransformer(nn.Module):
    """A decoder-only transformer whose layers are replaceable executors."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def replace_layer(self, index: int, executor: LayerExecutor) -> None:
        """Replace one block while preserving the model-level call contract."""

        if not isinstance(executor, LayerExecutor):
            raise TypeError("executor must implement LayerExecutor")
        self.layers[index] = executor

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        capture_activations: bool = False,
        retain_activation_gradients: bool = True,
        activation_interventions: (
            Mapping[str, ActivationIntervention] | None
        ) = None,
    ) -> TransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {input_ids.shape}"
            )
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same [batch, sequence] shape as input_ids"
            )
        else:
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        if not attention_mask[:, 0].all():
            raise ValueError(
                "attention_mask must describe right padding (the first token must be valid)"
            )

        needs_trace = capture_activations or bool(activation_interventions)
        trace = None
        if needs_trace:
            trace = ActivationTrace(
                retain_grad=(
                    retain_activation_gradients
                    if capture_activations
                    else False
                ),
                interventions=activation_interventions,
                store=capture_activations,
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        token_embeddings = record(
            trace, "embedding.token", self.token_embedding(input_ids)
        )
        position_embeddings = record(
            trace, "embedding.position", self.position_embedding(positions)
        )
        hidden_states = record(
            trace,
            "embedding.output",
            self.embedding_dropout(token_embeddings + position_embeddings),
        )

        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                trace=trace,
                prefix=f"layer.{index}",
            )
        hidden_states = record(trace, "final_norm", self.final_norm(hidden_states))
        logits = record(trace, "logits", self.lm_head(hidden_states))
        if trace is not None:
            trace.assert_all_interventions_applied()
        return TransformerOutput(
            logits=logits,
            activations=trace if capture_activations else None,
        )
