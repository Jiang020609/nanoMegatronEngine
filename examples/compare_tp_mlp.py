"""Compare a normal GPT MLP with the fake tensor-parallel MLP."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig
from nano_megatron_engine.model.mlp import MLP
from nano_megatron_engine.parallel import ColumnParallelLinear, RowParallelLinear
from nano_megatron_engine.parallel.fake_tp import split_tensor_along_dim


def main() -> None:
    torch.manual_seed(2027)
    normal_config = GPTConfig(n_embd=16, n_head=2, dropout=0.0, tensor_parallel_size=1)
    tp_config = GPTConfig(n_embd=16, n_head=2, dropout=0.0, tensor_parallel_size=2)
    normal_mlp = MLP(normal_config)
    tp_mlp = MLP(tp_config)
    copy_normal_mlp_to_tp_mlp(normal_mlp, tp_mlp)

    x = torch.randn(4, 8, normal_config.n_embd)
    normal_output = normal_mlp(x)
    tp_output = tp_mlp(x)
    max_error = (tp_output - normal_output).abs().max().item()

    print("Fake TP MLP comparison")
    print("ColumnParallelLinear shards the MLP expansion dimension.")
    print("RowParallelLinear sums partial outputs back to the model hidden size.")
    print("This is single-process educational fake TP, not distributed speedup.")
    print(f"max_abs_error={max_error:.3e}")


def copy_normal_mlp_to_tp_mlp(normal_mlp: MLP, tp_mlp: MLP) -> None:
    first_linear = normal_mlp.net[0]
    second_linear = normal_mlp.net[2]
    first_tp = tp_mlp.net[0]
    second_tp = tp_mlp.net[2]

    if not isinstance(first_linear, nn.Linear) or not isinstance(second_linear, nn.Linear):
        raise TypeError("normal_mlp must use nn.Linear layers")
    if not isinstance(first_tp, ColumnParallelLinear) or not isinstance(second_tp, RowParallelLinear):
        raise TypeError("tp_mlp must use fake tensor-parallel layers")

    with torch.no_grad():
        first_weight_chunks = split_tensor_along_dim(first_linear.weight, first_tp.tp_size, dim=0)
        for shard, chunk in zip(first_tp.weight_shards, first_weight_chunks):
            shard.copy_(chunk)
        if first_linear.bias is not None and first_tp.bias_shards is not None:
            first_bias_chunks = split_tensor_along_dim(first_linear.bias, first_tp.tp_size, dim=0)
            for shard, chunk in zip(first_tp.bias_shards, first_bias_chunks):
                shard.copy_(chunk)

        second_weight_chunks = split_tensor_along_dim(second_linear.weight, second_tp.tp_size, dim=1)
        for shard, chunk in zip(second_tp.weight_shards, second_weight_chunks):
            shard.copy_(chunk)
        if second_linear.bias is not None and second_tp.bias is not None:
            second_tp.bias.copy_(second_linear.bias)


if __name__ == "__main__":
    main()
