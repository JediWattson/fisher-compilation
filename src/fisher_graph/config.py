"""Configuration for the toy transformer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    vocab_size: int = 128
    max_sequence_length: int = 64
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.0
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

