"""Configuration for the tiny GPT model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Small GPT configuration.

    The defaults are intentionally modest so the model can run on CPU during
    tests and examples.
    """

    vocab_size: int = 256
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    use_activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "GPTConfig":
        """Build a config from a plain dictionary."""

        return cls(**values)

