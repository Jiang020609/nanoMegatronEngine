"""Feed-forward block used inside each transformer block."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.parallel import ColumnParallelLinear, RowParallelLinear


class MLP(nn.Module):
    """GPT-style MLP with a 4x hidden dimension."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_hidden_size
        if config.tensor_parallel_size == 1:
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, config.n_embd),
                nn.Dropout(config.dropout),
            )
        else:
            # Single-process fake TP: column parallel shards output features,
            # then row parallel shards input features and sums partial outputs.
            self.net = nn.Sequential(
                ColumnParallelLinear(
                    config.n_embd,
                    hidden_dim,
                    tp_size=config.tensor_parallel_size,
                    gather_output=True,
                ),
                nn.GELU(),
                RowParallelLinear(hidden_dim, config.n_embd, tp_size=config.tensor_parallel_size),
                nn.Dropout(config.dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
