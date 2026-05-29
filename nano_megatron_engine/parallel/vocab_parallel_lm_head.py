"""Vocab-parallel LM head modules."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import (
    DistributedRankLocalCollectives,
    FakeShardListCollectives,
    ShardListCollectiveProtocol,
)
from nano_megatron_engine.parallel.fake_tp import partition_range


class VocabParallelLMHead(nn.Module):
    """Split LM-head output vocab rows across fake or rank-local distributed shards."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tp_size: int = 2,
        bias: bool = False,
        collectives: ShardListCollectiveProtocol | DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")

        self.collectives = collectives if collectives is not None else FakeShardListCollectives()
        self.is_rank_local = isinstance(self.collectives, DistributedRankLocalCollectives)
        self.rank = self.collectives.get_rank() if self.is_rank_local else 0
        self.world_size = self.collectives.get_world_size() if self.is_rank_local else tp_size
        if self.is_rank_local and tp_size != self.world_size:
            raise ValueError(f"distributed vocab LM head tp_size={tp_size} must match world_size={self.world_size}")
        if self.is_rank_local and vocab_size % self.world_size != 0:
            raise ValueError(
                "distributed vocab LM head currently requires divisible vocab partitions, "
                f"got vocab_size={vocab_size}, world_size={self.world_size}"
            )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = self.world_size if self.is_rank_local else tp_size
        self.vocab_ranges = [partition_range(vocab_size, self.tp_size, idx) for idx in range(self.tp_size)]
        self.local_vocab_start, self.local_vocab_end = self.vocab_ranges[self.rank]
        parameter_ranges = [self.vocab_ranges[self.rank]] if self.is_rank_local else self.vocab_ranges
        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(end - start, hidden_size)) for start, end in parameter_ranges]
        )
        self.bias_shards = (
            nn.ParameterList([nn.Parameter(torch.empty(end - start)) for start, end in parameter_ranges])
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
        collectives: ShardListCollectiveProtocol | DistributedRankLocalCollectives | None = None,
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
            if layer.is_rank_local:
                start, end = layer.local_vocab_start, layer.local_vocab_end
                layer.weight_shards[0].copy_(linear.weight[start:end])
                if linear.bias is not None and layer.bias_shards is not None:
                    layer.bias_shards[0].copy_(linear.bias[start:end])
            else:
                for weight_shard, (start, end) in zip(layer.weight_shards, layer.vocab_ranges):
                    weight_shard.copy_(linear.weight[start:end])
                if linear.bias is not None and layer.bias_shards is not None:
                    for bias_shard, (start, end) in zip(layer.bias_shards, layer.vocab_ranges):
                        bias_shard.copy_(linear.bias[start:end])
        return layer

    def tie_weight_shards(self, weight_shards: nn.ParameterList) -> None:
        """Share vocab shard parameters with a vocab-parallel embedding."""

        expected_shards = 1 if self.is_rank_local else self.tp_size
        if len(weight_shards) != expected_shards:
            message = (
                "weight_shards length must match rank-local shard count"
                if self.is_rank_local
                else "weight_shards length must match tp_size"
            )
            raise ValueError(message)
        ranges = [self.vocab_ranges[self.rank]] if self.is_rank_local else self.vocab_ranges
        for weight, (start, end) in zip(weight_shards, ranges):
            expected_shape = (end - start, self.hidden_size)
            if tuple(weight.shape) != expected_shape:
                raise ValueError(f"weight shard shape {tuple(weight.shape)} does not match {expected_shape}")
        self.weight_shards = weight_shards

    def merge_to_linear(self) -> nn.Linear:
        if self.is_rank_local:
            weight = self.collectives.all_gather(self.weight_shards[0], dim=0)
        else:
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
                if self.is_rank_local:
                    linear.bias.copy_(self.collectives.all_gather(self.bias_shards[0], dim=0))
                else:
                    linear.bias.copy_(self.collectives.all_gather(list(self.bias_shards), dim=0))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_rank_local:
            if x.device.type != "cpu":
                raise ValueError(
                    "VocabParallelLMHead distributed path supports CPU/Gloo tensors only, "
                    f"got {x.device}"
                )
            bias = self.bias_shards[0] if self.bias_shards is not None else None
            return _DistributedVocabParallelLMHeadFunction.apply(
                x,
                self.weight_shards[0],
                bias,
                self.local_vocab_start,
                self.local_vocab_end,
                self.collectives.group,
            )

        local_logits = tuple(
            F.linear(x, weight, bias)
            for weight, bias in zip(self.weight_shards, self._iter_bias_shards())
        )
        return self.collectives.all_gather(local_logits, dim=-1)

    def extra_repr(self) -> str:
        mode = "distributed_rank_local" if self.is_rank_local else "fake_shard_list"
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"tp_size={self.tp_size}, bias={self.bias_shards is not None}, mode={mode}"
        )

    def _iter_bias_shards(self) -> tuple[torch.Tensor | None, ...]:
        if self.bias_shards is None:
            return tuple(None for _ in range(self.tp_size))
        return tuple(self.bias_shards)


class _DistributedVocabParallelLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        local_vocab_start: int,
        local_vocab_end: int,
        group: object | None,
    ) -> torch.Tensor:
        from nano_megatron_engine.parallel.distributed_collectives import distributed_all_gather

        local_logits = F.linear(x, weight, bias)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.local_vocab_start = local_vocab_start
        ctx.local_vocab_end = local_vocab_end
        ctx.group = group

        return distributed_all_gather(local_logits, dim=-1, group=group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as dist

        x, weight = ctx.saved_tensors
        grad_local_logits = grad_output[..., ctx.local_vocab_start : ctx.local_vocab_end].contiguous()
        grad_input = F.linear(grad_local_logits, weight.t())
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_logits_2d = grad_local_logits.reshape(-1, grad_local_logits.shape[-1])
        grad_weight = grad_logits_2d.t().matmul(x_2d)
        grad_bias = _sum_bias_gradient(grad_local_logits) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None


def _sum_bias_gradient(grad_output: torch.Tensor) -> torch.Tensor:
    if grad_output.ndim == 1:
        return grad_output
    return grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
