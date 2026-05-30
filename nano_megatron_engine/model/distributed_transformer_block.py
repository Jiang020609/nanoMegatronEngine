"""CPU/Gloo distributed transformer block prototype."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model.distributed_attention import DistributedCausalSelfAttention
from nano_megatron_engine.parallel import DistributedColumnParallelLinear, DistributedRowParallelLinear
from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedTransformerBlock(nn.Module):
    """Rank-local CPU/Gloo transformer block prototype.

    LayerNorm parameters are replicated on every rank. Attention uses
    ``DistributedCausalSelfAttention`` and the MLP composes column-parallel and
    row-parallel linear prototypes. This is not wired into ``GPTModel``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        mlp_hidden_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        collectives: DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"distributed transformer block hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"distributed transformer block num_heads must be positive, got {num_heads}")
        if block_size <= 0:
            raise ValueError(f"distributed transformer block block_size must be positive, got {block_size}")
        if mlp_hidden_size <= 0:
            raise ValueError(
                f"distributed transformer block mlp_hidden_size must be positive, got {mlp_hidden_size}"
            )
        if not isinstance(bias, bool):
            raise TypeError(f"distributed transformer block bias must be bool, got {type(bias).__name__}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                "distributed transformer block requires "
                f"hidden_size={hidden_size} to be divisible by num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"distributed transformer block dropout must be in [0.0, 1.0), got {dropout}")

        self.collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
        self.rank = self.collectives.get_rank()
        self.world_size = self.collectives.get_world_size()
        if self.world_size <= 0:
            raise ValueError(f"distributed transformer block world_size must be positive, got {self.world_size}")
        if num_heads % self.world_size != 0:
            raise ValueError(
                "distributed transformer block requires strict head divisibility: "
                f"num_heads={num_heads} must be divisible by world_size={self.world_size}"
            )
        if mlp_hidden_size % self.world_size != 0:
            raise ValueError(
                "distributed transformer block requires "
                f"mlp_hidden_size={mlp_hidden_size} to be divisible by world_size={self.world_size}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout_p = dropout
        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // self.world_size

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = DistributedCausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            block_size=block_size,
            bias=bias,
            dropout=dropout,
            collectives=self.collectives,
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.fc1 = DistributedColumnParallelLinear(
            in_features=hidden_size,
            out_features=mlp_hidden_size,
            bias=bias,
            gather_output=False,
            collectives=self.collectives,
        )
        self.activation = nn.GELU()
        self.fc2 = DistributedRowParallelLinear(
            in_features=mlp_hidden_size,
            out_features=hidden_size,
            bias=bias,
            input_is_parallel=True,
            collectives=self.collectives,
        )
        self.mlp_dropout = nn.Dropout(dropout)

    def copy_from_dense_(self, dense_block: nn.Module) -> "DistributedTransformerBlock":
        """Copy replicated LayerNorms and rank-local attention/MLP shards."""

        dense_ln_1 = getattr(dense_block, "ln_1", None)
        dense_attn = getattr(dense_block, "attn", None)
        dense_ln_2 = getattr(dense_block, "ln_2", None)
        dense_mlp = getattr(dense_block, "mlp", None)
        dense_mlp_net = getattr(dense_mlp, "net", None)

        if not isinstance(dense_ln_1, nn.LayerNorm) or not isinstance(dense_ln_2, nn.LayerNorm):
            raise TypeError("dense transformer block must expose nn.LayerNorm fields named ln_1 and ln_2")
        if not isinstance(dense_mlp_net, nn.Sequential) or len(dense_mlp_net) < 3:
            raise TypeError("dense transformer block MLP must expose a net Sequential with Linear layers at 0 and 2")
        dense_fc1 = dense_mlp_net[0]
        dense_fc2 = dense_mlp_net[2]
        if not isinstance(dense_fc1, nn.Linear) or not isinstance(dense_fc2, nn.Linear):
            raise TypeError("dense transformer block MLP net[0] and net[2] must be nn.Linear")

        with torch.no_grad():
            self.ln_1.weight.copy_(dense_ln_1.weight)
            self.ln_1.bias.copy_(dense_ln_1.bias)
            self.ln_2.weight.copy_(dense_ln_2.weight)
            self.ln_2.bias.copy_(dense_ln_2.bias)

        self.attn.copy_from_dense_(dense_attn)
        self.fc1.copy_from_dense_(dense_fc1)
        self.fc2.copy_from_dense_(dense_fc2)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"distributed transformer block expected a torch.Tensor input, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError("distributed transformer block input must have shape [batch, seq, hidden_size]")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"distributed transformer block expected hidden_size={self.hidden_size}, got {x.shape[-1]}"
            )
        if x.shape[1] > self.block_size:
            raise ValueError(
                f"distributed transformer block sequence length {x.shape[1]} exceeds block_size {self.block_size}"
            )
        if x.device.type != "cpu":
            raise ValueError(f"DistributedTransformerBlock currently supports CPU/Gloo tensors only, got {x.device}")

        x = x + self.attn(self.ln_1(x))
        local_hidden = self.fc1(self.ln_2(x))
        local_hidden = self.activation(local_hidden)
        x = x + self.mlp_dropout(self.fc2(local_hidden))
        return x

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"local_heads={self.local_heads}, mlp_hidden_size={self.mlp_hidden_size}, "
            f"block_size={self.block_size}, rank={self.rank}, world_size={self.world_size}, "
            f"dropout={self.dropout_p}"
        )
