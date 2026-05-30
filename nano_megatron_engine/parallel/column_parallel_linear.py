"""Fake Megatron-style column-parallel linear layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import (
    FakeShardListCollectives,
    ShardListCollectiveProtocol,
)
from nano_megatron_engine.parallel.fake_tp import (
    split_tensor_along_dim,
    validate_divisible,
)


class ColumnParallelLinear(nn.Module):
    """Split nn.Linear's output dimension across local fake TP shards.

    A standard linear layer stores weight as [out_features, in_features].
    Column parallelism shards that weight along dim=0, so each shard owns a
    contiguous slice of output features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 2,
        bias: bool = True,
        gather_output: bool = True,
        collectives: ShardListCollectiveProtocol | None = None,
    ) -> None:
        super().__init__()
        validate_divisible(out_features, tp_size, "out_features")
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.collectives = collectives if collectives is not None else FakeShardListCollectives()
        self.local_out_features = out_features // tp_size

        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(self.local_out_features, in_features)) for _ in range(tp_size)]
        )
        self.bias_shards = (
            nn.ParameterList([nn.Parameter(torch.empty(self.local_out_features)) for _ in range(tp_size)])
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight_shards:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias_shards is not None:
            bound = 1 / math.sqrt(self.in_features)
            for bias in self.bias_shards:
                nn.init.uniform_(bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tp_size: int = 2,
        gather_output: bool = True,
        collectives: ShardListCollectiveProtocol | None = None,
    ) -> "ColumnParallelLinear":
        """Create a column-parallel layer with weights copied from nn.Linear."""

        layer = cls(
            linear.in_features,
            linear.out_features,
            tp_size=tp_size,
            bias=linear.bias is not None,
            gather_output=gather_output,
            collectives=collectives,
        )
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)

        with torch.no_grad():
            weight_chunks = split_tensor_along_dim(linear.weight.detach(), tp_size, dim=0)
            for shard, chunk in zip(layer.weight_shards, weight_chunks):
                shard.copy_(chunk)
            if linear.bias is not None and layer.bias_shards is not None:
                bias_chunks = split_tensor_along_dim(linear.bias.detach(), tp_size, dim=0)
                for shard, chunk in zip(layer.bias_shards, bias_chunks):
                    shard.copy_(chunk)
        return layer

    def merge_to_linear(self) -> nn.Linear:
        """Merge local fake TP shards back into an equivalent nn.Linear."""

        first_weight = self.weight_shards[0]
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias_shards is not None,
            device=first_weight.device,
            dtype=first_weight.dtype,
        )

        with torch.no_grad():
            linear.weight.copy_(self.collectives.all_gather(list(self.weight_shards), dim=0))
            if self.bias_shards is not None and linear.bias is not None:
                linear.bias.copy_(self.collectives.all_gather(list(self.bias_shards), dim=0))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input last dimension {self.in_features}, got {x.shape[-1]}")

        local_outputs = tuple(
            F.linear(x, weight, bias)
            for weight, bias in zip(self.weight_shards, self._iter_bias_shards())
        )
        if self.gather_output:
            return self.collectives.all_gather(local_outputs, dim=-1)
        return local_outputs

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tp_size={self.tp_size}, bias={self.bias_shards is not None}, "
            f"gather_output={self.gather_output}"
        )

    def _iter_bias_shards(self) -> tuple[torch.Tensor | None, ...]:
        if self.bias_shards is None:
            return tuple(None for _ in range(self.tp_size))
        return tuple(self.bias_shards)
