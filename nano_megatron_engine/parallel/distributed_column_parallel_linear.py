"""Rank-local distributed column-parallel linear prototype."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedColumnParallelLinear(nn.Module):
    """Shard a linear layer's output features across distributed ranks.

    This is a low-level prototype. Each process owns one rank-local weight
    shard and receives the full input tensor. It is not wired into the GPT model
    path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
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
        if out_features % self.world_size != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by distributed world_size={self.world_size}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.local_out_features = out_features // self.world_size
        self.local_out_start = self.rank * self.local_out_features
        self.local_out_end = self.local_out_start + self.local_out_features

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def copy_from_dense_(self, dense: nn.Linear) -> None:
        """Copy this rank's output-feature shard from a dense nn.Linear."""

        if not isinstance(dense, nn.Linear):
            raise TypeError(f"dense must be an nn.Linear, got {type(dense).__name__}")
        if dense.in_features != self.in_features or dense.out_features != self.out_features:
            raise ValueError(
                "dense Linear shape must match distributed layer shape, "
                f"got in={dense.in_features}, out={dense.out_features}; "
                f"expected in={self.in_features}, out={self.out_features}"
            )
        if (dense.bias is None) != (self.bias is None):
            raise ValueError("dense bias setting must match distributed layer bias setting")

        with torch.no_grad():
            self.weight.copy_(dense.weight[self.local_out_start : self.local_out_end])
            if self.bias is not None and dense.bias is not None:
                self.bias.copy_(dense.bias[self.local_out_start : self.local_out_end])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input last dimension {self.in_features}, got {x.shape[-1]}")
        self.collectives.validate_tensor_device(x, "DistributedColumnParallelLinear")

        return _DistributedColumnParallelLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.gather_output,
            self.local_out_start,
            self.local_out_end,
            self.collectives.group,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, rank={self.rank}, "
            f"world_size={self.world_size}, bias={self.bias is not None}, "
            f"gather_output={self.gather_output}"
        )


class _DistributedColumnParallelLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gather_output: bool,
        local_out_start: int,
        local_out_end: int,
        group: object | None,
    ) -> torch.Tensor:
        import torch.distributed as dist

        local_output = F.linear(x, weight, bias)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.gather_output = gather_output
        ctx.local_out_start = local_out_start
        ctx.local_out_end = local_out_end
        ctx.group = group

        if not gather_output:
            return local_output

        gathered = [torch.empty_like(local_output) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(gathered, local_output.contiguous(), group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as dist

        x, weight = ctx.saved_tensors
        if ctx.gather_output:
            grad_local_output = grad_output[..., ctx.local_out_start : ctx.local_out_end].contiguous()
        else:
            grad_local_output = grad_output.contiguous()

        grad_input = F.linear(grad_local_output, weight.t())
        if ctx.gather_output:
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_output_2d = grad_local_output.reshape(-1, grad_local_output.shape[-1])
        grad_weight = grad_output_2d.t().matmul(x_2d)
        grad_bias = _sum_bias_gradient(grad_local_output) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None


def _sum_bias_gradient(grad_output: torch.Tensor) -> torch.Tensor:
    if grad_output.ndim == 1:
        return grad_output
    return grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
