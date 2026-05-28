"""Causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.parallel import RowParallelLinear


class CausalSelfAttention(nn.Module):
    """Standard multi-head masked self-attention for autoregressive GPT."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.tensor_parallel_size = config.tensor_parallel_size
        self.local_heads = config.n_head // config.tensor_parallel_size
        self.local_hidden = self.local_heads * self.head_dim
        self.dropout_p = config.dropout

        if self.tensor_parallel_size == 1:
            self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.proj = nn.Linear(config.n_embd, config.n_embd)
        else:
            # Single-process fake TP: each shard owns local Q, K, and V heads.
            self.qkv_weight_shards = nn.ParameterList(
                [nn.Parameter(torch.empty(3 * self.local_hidden, config.n_embd)) for _ in range(self.tensor_parallel_size)]
            )
            self.qkv_bias_shards = nn.ParameterList(
                [nn.Parameter(torch.empty(3 * self.local_hidden)) for _ in range(self.tensor_parallel_size)]
            )
            self._reset_qkv_shards()
            # The output projection consumes local context slices and sums the
            # partial outputs, matching row-parallel semantics in one process.
            self.proj = RowParallelLinear(config.n_embd, config.n_embd, tp_size=self.tensor_parallel_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        if self.tensor_parallel_size == 1:
            qkv = self.qkv(x)
            query, key, value = qkv.split(self.n_embd, dim=2)
            query = self._shape_heads(query, batch_size, seq_len, self.n_head)
            key = self._shape_heads(key, batch_size, seq_len, self.n_head)
            value = self._shape_heads(value, batch_size, seq_len, self.n_head)
            y = self._apply_attention(query, key, value, seq_len)
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        else:
            local_contexts = []
            for weight, bias in zip(self.qkv_weight_shards, self.qkv_bias_shards):
                local_qkv = F.linear(x, weight, bias)
                query, key, value = local_qkv.split(self.local_hidden, dim=2)
                query = self._shape_heads(query, batch_size, seq_len, self.local_heads)
                key = self._shape_heads(key, batch_size, seq_len, self.local_heads)
                value = self._shape_heads(value, batch_size, seq_len, self.local_heads)
                local_y = self._apply_attention(query, key, value, seq_len)
                local_y = local_y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.local_hidden)
                local_contexts.append(local_y)
            y = torch.cat(local_contexts, dim=-1)

        y = self.resid_dropout(self.proj(y))
        return y

    def _apply_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        return weights @ value

    def _shape_heads(self, x: torch.Tensor, batch_size: int, seq_len: int, n_head: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, n_head, self.head_dim).transpose(1, 2)

    def _reset_qkv_shards(self) -> None:
        bound = 1 / math.sqrt(self.n_embd)
        for weight in self.qkv_weight_shards:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        for bias in self.qkv_bias_shards:
            nn.init.uniform_(bias, -bound, bound)
