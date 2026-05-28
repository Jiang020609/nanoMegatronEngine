import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig
from nano_megatron_engine.model.mlp import MLP
from nano_megatron_engine.parallel import ColumnParallelLinear, RowParallelLinear
from nano_megatron_engine.parallel.fake_tp import split_tensor_along_dim


def test_fake_tp_mlp_matches_normal_mlp_after_weight_copy():
    torch.manual_seed(333)
    normal_mlp = MLP(GPTConfig(n_embd=16, n_head=2, dropout=0.0, tensor_parallel_size=1))
    tp_mlp = MLP(GPTConfig(n_embd=16, n_head=2, dropout=0.0, tensor_parallel_size=2))
    copy_normal_mlp_to_tp_mlp(normal_mlp, tp_mlp)

    x = torch.randn(2, 5, 16)

    assert torch.allclose(tp_mlp(x), normal_mlp(x), atol=1e-6, rtol=1e-6)


def copy_normal_mlp_to_tp_mlp(normal_mlp: MLP, tp_mlp: MLP) -> None:
    first_linear = normal_mlp.net[0]
    second_linear = normal_mlp.net[2]
    first_tp = tp_mlp.net[0]
    second_tp = tp_mlp.net[2]

    assert isinstance(first_linear, nn.Linear)
    assert isinstance(second_linear, nn.Linear)
    assert isinstance(first_tp, ColumnParallelLinear)
    assert isinstance(second_tp, RowParallelLinear)

    with torch.no_grad():
        first_weight_chunks = split_tensor_along_dim(first_linear.weight, first_tp.tp_size, dim=0)
        for shard, chunk in zip(first_tp.weight_shards, first_weight_chunks):
            shard.copy_(chunk)
        first_bias_chunks = split_tensor_along_dim(first_linear.bias, first_tp.tp_size, dim=0)
        for shard, chunk in zip(first_tp.bias_shards, first_bias_chunks):
            shard.copy_(chunk)

        second_weight_chunks = split_tensor_along_dim(second_linear.weight, second_tp.tp_size, dim=1)
        for shard, chunk in zip(second_tp.weight_shards, second_weight_chunks):
            shard.copy_(chunk)
        second_tp.bias.copy_(second_linear.bias)
