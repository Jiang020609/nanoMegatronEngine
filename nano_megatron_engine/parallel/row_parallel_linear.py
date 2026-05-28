"""Fake Megatron-style row-parallel linear layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.fake_tp import (
    concat_tensor_parallel_outputs,
    split_tensor_along_dim,
    sum_tensor_parallel_outputs,
    validate_divisible,
)


class RowParallelLinear(nn.Module):
    """Split nn.Linear's input dimension across local fake TP shards.

    A standard linear layer stores weight as [out_features, in_features].
    Row parallelism shards that weight along dim=1, computes partial outputs,
    then sums those partial outputs and applies bias once.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        validate_divisible(in_features, tp_size, "in_features")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.local_in_features = in_features // tp_size

        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(out_features, self.local_in_features)) for _ in range(tp_size)]
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        for weight in self.weight_shards:
            nn.init.uniform_(weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, tp_size: int = 2) -> "RowParallelLinear":
        """Create a row-parallel layer with weights copied from nn.Linear."""

        layer = cls(
            linear.in_features,
            linear.out_features,
            tp_size=tp_size,
            bias=linear.bias is not None,
        )
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)

        with torch.no_grad():
            weight_chunks = split_tensor_along_dim(linear.weight.detach(), tp_size, dim=1)
            for shard, chunk in zip(layer.weight_shards, weight_chunks):
                shard.copy_(chunk)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias.detach())
        return layer

    def merge_to_linear(self) -> nn.Linear:
        """Merge local fake TP shards back into an equivalent nn.Linear."""

        first_weight = self.weight_shards[0]
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=first_weight.device,
            dtype=first_weight.dtype,
        )

        with torch.no_grad():
            linear.weight.copy_(concat_tensor_parallel_outputs(list(self.weight_shards), dim=1))
            if self.bias is not None and linear.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input last dimension {self.in_features}, got {x.shape[-1]}")

        input_shards = split_tensor_along_dim(x, self.tp_size, dim=-1)
        partial_outputs = tuple(
            F.linear(input_shard, weight, bias=None)
            for input_shard, weight in zip(input_shards, self.weight_shards)
        )
        output = sum_tensor_parallel_outputs(partial_outputs)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tp_size={self.tp_size}, bias={self.bias is not None}"
        )

