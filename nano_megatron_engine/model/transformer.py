"""Transformer block for the tiny GPT model."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.mlp import MLP


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

