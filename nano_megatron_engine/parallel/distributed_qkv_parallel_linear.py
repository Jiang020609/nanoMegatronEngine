"""Rank-local distributed QKV projection prototype."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedQKVParallelLinear(nn.Module):
    """Shard Q, K, and V projection rows by local attention heads.

    Dense GPT attention usually stores QKV projection rows as
    ``[Q_all, K_all, V_all]``. A plain column-parallel split over
    ``3 * hidden_size`` would not preserve that grouping, so this prototype
    gives each distributed rank its local Q heads, local K heads, and local V
    heads in ``[Q_local, K_local, V_local]`` order.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        bias: bool = True,
        collectives: DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"distributed QKV hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"distributed QKV num_heads must be positive, got {num_heads}")
        if not isinstance(bias, bool):
            raise TypeError(f"distributed QKV bias must be bool, got {type(bias).__name__}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"distributed QKV requires hidden_size={hidden_size} to be divisible by num_heads={num_heads}"
            )

        self.collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
        self.rank = self.collectives.get_rank()
        self.world_size = self.collectives.get_world_size()
        if self.world_size <= 0:
            raise ValueError(f"distributed QKV world_size must be positive, got {self.world_size}")
        if num_heads % self.world_size != 0:
            raise ValueError(
                "distributed QKV requires strict head divisibility: "
                f"num_heads={num_heads} must be divisible by world_size={self.world_size} "
                "so every rank owns an integer number of local heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // self.world_size
        self.local_hidden = self.local_heads * self.head_dim
        self.local_start = self.rank * self.local_hidden
        self.local_end = self.local_start + self.local_hidden

        self.weight = nn.Parameter(torch.empty(3 * self.local_hidden, hidden_size))
        self.bias = nn.Parameter(torch.empty(3 * self.local_hidden)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.hidden_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def copy_from_dense_(self, dense: nn.Linear) -> "DistributedQKVParallelLinear":
        """Copy this rank's Q, K, and V head rows from a dense QKV Linear."""

        if not isinstance(dense, nn.Linear):
            raise TypeError(f"dense must be an nn.Linear, got {type(dense).__name__}")
        expected_out_features = 3 * self.hidden_size
        if dense.in_features != self.hidden_size or dense.out_features != expected_out_features:
            raise ValueError(
                "dense QKV Linear shape must match distributed QKV shape, "
                f"got in={dense.in_features}, out={dense.out_features}; "
                f"expected in={self.hidden_size}, out={expected_out_features}"
            )
        if (dense.bias is None) != (self.bias is None):
            raise ValueError("dense QKV bias setting must match distributed QKV bias setting")

        q_start, q_end = self.local_start, self.local_end
        k_start, k_end = self.hidden_size + self.local_start, self.hidden_size + self.local_end
        v_start, v_end = 2 * self.hidden_size + self.local_start, 2 * self.hidden_size + self.local_end

        with torch.no_grad():
            self.weight.copy_(
                torch.cat(
                    [
                        dense.weight[q_start:q_end],
                        dense.weight[k_start:k_end],
                        dense.weight[v_start:v_end],
                    ],
                    dim=0,
                )
            )
            if self.bias is not None and dense.bias is not None:
                self.bias.copy_(
                    torch.cat(
                        [
                            dense.bias[q_start:q_end],
                            dense.bias[k_start:k_end],
                            dense.bias[v_start:v_end],
                        ],
                        dim=0,
                    )
                )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"distributed QKV expected a torch.Tensor input, got {type(x).__name__}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"distributed QKV expected input last dimension hidden_size={self.hidden_size}, "
                f"got {x.shape[-1]}"
            )
        self.collectives.validate_tensor_device(x, "DistributedQKVParallelLinear")
        return _DistributedQKVParallelLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.collectives.group,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"local_heads={self.local_heads}, head_dim={self.head_dim}, "
            f"rank={self.rank}, world_size={self.world_size}, bias={self.bias is not None}"
        )


class _DistributedQKVParallelLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group: object | None,
    ) -> torch.Tensor:
        local_output = F.linear(x, weight, bias)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.group = group
        return local_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as dist

        x, weight = ctx.saved_tensors
        grad_local_output = grad_output.contiguous()
        grad_input = F.linear(grad_local_output, weight.t())
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_output_2d = grad_local_output.reshape(-1, grad_local_output.shape[-1])
        grad_weight = grad_output_2d.t().matmul(x_2d)
        grad_bias = _sum_bias_gradient(grad_local_output) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


def _sum_bias_gradient(grad_output: torch.Tensor) -> torch.Tensor:
    if grad_output.ndim == 1:
        return grad_output
    return grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
