"""Rank-local distributed causal self-attention prototype."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel import (
    DistributedQKVParallelLinear,
    DistributedRowParallelLinear,
    RNGStateTracker,
    TrackedDropout,
)
from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedCausalSelfAttention(nn.Module):
    """Rank-local tensor-parallel causal self-attention.

    This module is an isolated prototype for learning distributed attention
    mechanics. Each rank owns local Q/K/V heads, computes attention for those
    heads, and uses row-parallel output projection to sum partial outputs. It is
    not wired into the GPT model path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        collectives: DistributedRankLocalCollectives | None = None,
        rng_tracker: RNGStateTracker | None = None,
        attn_dropout_rng_name: str = "attention_dropout",
        resid_dropout_rng_name: str = "residual_dropout",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"distributed causal attention hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"distributed causal attention num_heads must be positive, got {num_heads}")
        if block_size <= 0:
            raise ValueError(f"distributed causal attention block_size must be positive, got {block_size}")
        if not isinstance(bias, bool):
            raise TypeError(f"distributed causal attention bias must be bool, got {type(bias).__name__}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                "distributed causal attention requires "
                f"hidden_size={hidden_size} to be divisible by num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"distributed causal attention dropout must be in [0.0, 1.0), got {dropout}")

        self.collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
        self.rank = self.collectives.get_rank()
        self.world_size = self.collectives.get_world_size()
        if self.world_size <= 0:
            raise ValueError(f"distributed causal attention world_size must be positive, got {self.world_size}")
        if num_heads % self.world_size != 0:
            raise ValueError(
                "distributed causal attention requires strict head divisibility: "
                f"num_heads={num_heads} must be divisible by world_size={self.world_size} "
                "so every rank owns an integer number of local_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_p = dropout
        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // self.world_size
        self.local_hidden = self.local_heads * self.head_dim

        self.qkv = DistributedQKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            bias=bias,
            collectives=self.collectives,
        )
        self.proj = DistributedRowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            input_is_parallel=True,
            collectives=self.collectives,
        )
        self.attn_dropout = TrackedDropout(
            dropout,
            rng_tracker=rng_tracker,
            rng_name=attn_dropout_rng_name,
        )
        self.resid_dropout = TrackedDropout(
            dropout,
            rng_tracker=rng_tracker,
            rng_name=resid_dropout_rng_name,
        )

        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def copy_from_dense_(self, dense_attention: nn.Module) -> "DistributedCausalSelfAttention":
        """Copy QKV and output projection weights from a dense attention module."""

        dense_qkv = getattr(dense_attention, "qkv", None)
        dense_proj = getattr(dense_attention, "proj", None)
        if not isinstance(dense_qkv, nn.Linear) or not isinstance(dense_proj, nn.Linear):
            raise TypeError("dense attention must expose nn.Linear fields named qkv and proj")

        self.qkv.copy_from_dense_(dense_qkv)
        self.proj.copy_from_dense_(dense_proj)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"distributed causal attention expected a torch.Tensor input, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError("distributed causal attention input must have shape [batch, seq, hidden_size]")
        batch_size, seq_len, channels = x.shape
        if channels != self.hidden_size:
            raise ValueError(
                f"distributed causal attention expected hidden_size={self.hidden_size}, got {channels}"
            )
        if seq_len > self.block_size:
            raise ValueError(
                f"distributed causal attention sequence length {seq_len} exceeds block_size {self.block_size}"
            )
        self.collectives.validate_tensor_device(x, "DistributedCausalSelfAttention")

        local_qkv = self.qkv(x)
        query, key, value = local_qkv.split(self.local_hidden, dim=-1)
        query = self._shape_local_heads(query, batch_size, seq_len)
        key = self._shape_local_heads(key, batch_size, seq_len)
        value = self._shape_local_heads(value, batch_size, seq_len)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.local_hidden)
        output = self.proj(context)
        return self.resid_dropout(output)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"local_heads={self.local_heads}, head_dim={self.head_dim}, "
            f"block_size={self.block_size}, rank={self.rank}, world_size={self.world_size}, "
            f"dropout={self.dropout_p}, bias={self.qkv.bias is not None}"
        )

    def _shape_local_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
