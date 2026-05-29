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
    tensor_parallel_size: int = 1

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
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.tensor_parallel_size > 1 and self.mlp_hidden_size % self.tensor_parallel_size != 0:
            raise ValueError(
                "MLP intermediate_size="
                f"{self.mlp_hidden_size} must be divisible by tensor_parallel_size="
                f"{self.tensor_parallel_size}"
            )
        if self.tensor_parallel_size > 1 and self.n_head % self.tensor_parallel_size != 0:
            raise ValueError(
                f"n_head={self.n_head} must be divisible by tensor_parallel_size="
                f"{self.tensor_parallel_size}"
            )

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "GPTConfig":
        """Build a config from a plain dictionary."""

        return cls(**values)

    @property
    def mlp_hidden_size(self) -> int:
        """The 4x intermediate dimension used by the GPT MLP."""

        return 4 * self.n_embd
