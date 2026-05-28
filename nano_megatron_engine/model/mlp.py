"""Feed-forward block used inside each transformer block."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model.config import GPTConfig


class MLP(nn.Module):
    """GPT-style MLP with a 4x hidden dimension."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

