"""Rank-local distributed row-parallel linear prototype."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedRowParallelLinear(nn.Module):
    """Shard a linear layer's input features across distributed ranks.

    Each rank owns a contiguous slice of the input feature dimension and the
    matching local weight columns. Partial outputs are summed across ranks, and
    the optional replicated bias is applied once after that all-reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        collectives: DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        self.collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
        self.rank = self.collectives.get_rank()
        self.world_size = self.collectives.get_world_size()
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if in_features % self.world_size != 0:
            raise ValueError(
                f"row-parallel in_features={in_features} must be divisible by distributed world_size={self.world_size}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.local_in_features = in_features // self.world_size
        self.local_in_start = self.rank * self.local_in_features
        self.local_in_end = self.local_in_start + self.local_in_features

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def copy_from_dense_(self, dense: nn.Linear) -> None:
        """Copy this rank's input-feature shard from a dense nn.Linear."""

        if not isinstance(dense, nn.Linear):
            raise TypeError(f"dense must be an nn.Linear, got {type(dense).__name__}")
        if dense.in_features != self.in_features or dense.out_features != self.out_features:
            raise ValueError(
                "dense Linear shape must match distributed row-parallel layer shape, "
                f"got in={dense.in_features}, out={dense.out_features}; "
                f"expected in={self.in_features}, out={self.out_features}"
            )
        if (dense.bias is None) != (self.bias is None):
            raise ValueError("dense bias setting must match distributed row-parallel layer bias setting")

        with torch.no_grad():
            self.weight.copy_(dense.weight[:, self.local_in_start : self.local_in_end])
            if self.bias is not None and dense.bias is not None:
                self.bias.copy_(dense.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_features = self.local_in_features if self.input_is_parallel else self.in_features
        if x.shape[-1] != expected_features:
            mode = "input_is_parallel=True" if self.input_is_parallel else "input_is_parallel=False"
            raise ValueError(
                f"DistributedRowParallelLinear with {mode} expected input last dimension "
                f"{expected_features}, got {x.shape[-1]}"
            )
        self.collectives.validate_tensor_device(x, "DistributedRowParallelLinear")

        local_x = x if self.input_is_parallel else x[..., self.local_in_start : self.local_in_end]
        partial_output = F.linear(local_x, self.weight, bias=None)
        output = self.collectives.all_reduce_sum(partial_output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_in_features={self.local_in_features}, rank={self.rank}, "
            f"world_size={self.world_size}, bias={self.bias is not None}, "
            f"input_is_parallel={self.input_is_parallel}"
        )
