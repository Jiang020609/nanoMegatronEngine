"""Fake vocab-parallel LM head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import (
    FakeShardListCollectives,
    ShardListCollectiveProtocol,
)
from nano_megatron_engine.parallel.fake_tp import partition_range


class VocabParallelLMHead(nn.Module):
    """Split LM-head output vocab rows across fake TP shards."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tp_size: int = 2,
        bias: bool = False,
        collectives: ShardListCollectiveProtocol | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = tp_size
        self.collectives = collectives if collectives is not None else FakeShardListCollectives()
        self.vocab_ranges = [partition_range(vocab_size, tp_size, idx) for idx in range(tp_size)]
        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(end - start, hidden_size)) for start, end in self.vocab_ranges]
        )
        self.bias_shards = (
            nn.ParameterList([nn.Parameter(torch.empty(end - start)) for start, end in self.vocab_ranges])
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.hidden_size)
        for weight in self.weight_shards:
            nn.init.uniform_(weight, -bound, bound)
        if self.bias_shards is not None:
            for bias in self.bias_shards:
                nn.init.uniform_(bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tp_size: int = 2,
        collectives: ShardListCollectiveProtocol | None = None,
    ) -> "VocabParallelLMHead":
        layer = cls(
            hidden_size=linear.in_features,
            vocab_size=linear.out_features,
            tp_size=tp_size,
            bias=linear.bias is not None,
            collectives=collectives,
        )
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            for weight_shard, (start, end) in zip(layer.weight_shards, layer.vocab_ranges):
                weight_shard.copy_(linear.weight[start:end])
            if linear.bias is not None and layer.bias_shards is not None:
                for bias_shard, (start, end) in zip(layer.bias_shards, layer.vocab_ranges):
                    bias_shard.copy_(linear.bias[start:end])
        return layer

    def tie_weight_shards(self, weight_shards: nn.ParameterList) -> None:
        """Share vocab shard parameters with a vocab-parallel embedding."""

        if len(weight_shards) != self.tp_size:
            raise ValueError("weight_shards length must match tp_size")
        for weight, (start, end) in zip(weight_shards, self.vocab_ranges):
            expected_shape = (end - start, self.hidden_size)
            if tuple(weight.shape) != expected_shape:
                raise ValueError(f"weight shard shape {tuple(weight.shape)} does not match {expected_shape}")
        self.weight_shards = weight_shards

    def merge_to_linear(self) -> nn.Linear:
        weight = self.collectives.all_gather(list(self.weight_shards), dim=0)
        linear = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=self.bias_shards is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
            if self.bias_shards is not None and linear.bias is not None:
                linear.bias.copy_(self.collectives.all_gather(list(self.bias_shards), dim=0))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_logits = tuple(
            F.linear(x, weight, bias)
            for weight, bias in zip(self.weight_shards, self._iter_bias_shards())
        )
        return self.collectives.all_gather(local_logits, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"tp_size={self.tp_size}, bias={self.bias_shards is not None}"
        )

    def _iter_bias_shards(self) -> tuple[torch.Tensor | None, ...]:
        if self.bias_shards is None:
            return tuple(None for _ in range(self.tp_size))
        return tuple(self.bias_shards)
