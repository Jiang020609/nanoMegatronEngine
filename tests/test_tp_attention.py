import pytest
import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig
from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.parallel import RowParallelLinear
from nano_megatron_engine.parallel.fake_tp import split_tensor_along_dim


def test_attention_tensor_parallel_size_one_matches_dense_baseline():
    torch.manual_seed(440)
    config = GPTConfig(n_embd=16, n_head=4, block_size=8, dropout=0.0, tensor_parallel_size=1)
    dense = CausalSelfAttention(config)
    baseline = CausalSelfAttention(config)
    baseline.load_state_dict(dense.state_dict())
    dense.eval()
    baseline.eval()
    x = torch.randn(2, 8, config.n_embd)

    torch.testing.assert_close(dense(x), baseline(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tensor_parallel_size", [2, 4])
def test_fake_tp_attention_matches_dense_attention(tensor_parallel_size):
    torch.manual_seed(441 + tensor_parallel_size)
    dense_config = GPTConfig(n_embd=16, n_head=4, block_size=8, dropout=0.0, tensor_parallel_size=1)
    tp_config = GPTConfig(
        n_embd=16,
        n_head=4,
        block_size=8,
        dropout=0.0,
        tensor_parallel_size=tensor_parallel_size,
    )
    dense = CausalSelfAttention(dense_config)
    tp_attention = CausalSelfAttention(tp_config)
    copy_dense_attention_to_tp_attention(dense, tp_attention)
    dense.eval()
    tp_attention.eval()
    x = torch.randn(2, 8, dense_config.n_embd)

    dense_output = dense(x)
    tp_output = tp_attention(x)

    assert tp_output.shape == dense_output.shape
    torch.testing.assert_close(tp_output, dense_output, atol=1e-6, rtol=1e-5)


def test_invalid_attention_head_tensor_parallel_combination_raises():
    with pytest.raises(ValueError, match="n_head=3 must be divisible by tensor_parallel_size=2"):
        GPTConfig(n_embd=24, n_head=3, tensor_parallel_size=2)


def copy_dense_attention_to_tp_attention(dense: CausalSelfAttention, tp_attention: CausalSelfAttention) -> None:
    assert isinstance(dense.qkv, nn.Linear)
    assert isinstance(dense.proj, nn.Linear)
    assert isinstance(tp_attention.proj, RowParallelLinear)

    hidden_size = dense.n_embd
    local_hidden = tp_attention.local_hidden

    with torch.no_grad():
        for shard_idx, (weight_shard, bias_shard) in enumerate(
            zip(tp_attention.qkv_weight_shards, tp_attention.qkv_bias_shards)
        ):
            q_start = shard_idx * local_hidden
            q_end = (shard_idx + 1) * local_hidden
            k_start = hidden_size + q_start
            k_end = hidden_size + q_end
            v_start = 2 * hidden_size + q_start
            v_end = 2 * hidden_size + q_end

            weight_shard.copy_(
                torch.cat(
                    [
                        dense.qkv.weight[q_start:q_end],
                        dense.qkv.weight[k_start:k_end],
                        dense.qkv.weight[v_start:v_end],
                    ],
                    dim=0,
                )
            )
            bias_shard.copy_(
                torch.cat(
                    [
                        dense.qkv.bias[q_start:q_end],
                        dense.qkv.bias[k_start:k_end],
                        dense.qkv.bias[v_start:v_end],
                    ],
                    dim=0,
                )
            )

        proj_weight_chunks = split_tensor_along_dim(dense.proj.weight, tp_attention.proj.tp_size, dim=1)
        for shard, chunk in zip(tp_attention.proj.weight_shards, proj_weight_chunks):
            shard.copy_(chunk)
        tp_attention.proj.bias.copy_(dense.proj.bias)
