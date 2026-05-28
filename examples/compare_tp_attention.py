"""Compare dense GPT attention with fake tensor-parallel attention."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig
from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.parallel import RowParallelLinear
from nano_megatron_engine.parallel.fake_tp import split_tensor_along_dim


def main() -> None:
    torch.manual_seed(2028)
    dense_config = GPTConfig(n_embd=16, n_head=4, block_size=8, dropout=0.0, tensor_parallel_size=1)
    tp_config = GPTConfig(n_embd=16, n_head=4, block_size=8, dropout=0.0, tensor_parallel_size=2)
    dense = CausalSelfAttention(dense_config)
    tp_attention = CausalSelfAttention(tp_config)
    copy_dense_attention_to_tp_attention(dense, tp_attention)
    dense.eval()
    tp_attention.eval()

    x = torch.randn(2, dense_config.block_size, dense_config.n_embd)
    dense_output = dense(x)
    tp_output = tp_attention(x)
    max_error = (tp_output - dense_output).abs().max().item()
    close = torch.allclose(tp_output, dense_output, atol=1e-6, rtol=1e-5)

    print("Fake TP attention comparison")
    print("This is single-process educational fake TP, not distributed communication.")
    print(f"tensor_parallel_size={tp_config.tensor_parallel_size}")
    print(f"num_heads={tp_config.n_head}")
    print(f"local_heads={tp_attention.local_heads}")
    print(f"head_dim={tp_attention.head_dim}")
    print(f"dense_output_shape={tuple(dense_output.shape)}")
    print(f"tp_output_shape={tuple(tp_output.shape)}")
    print(f"qkv_shard_shape={tuple(tp_attention.qkv_weight_shards[0].shape)}")
    print(f"proj_weight_shard_shape={tuple(tp_attention.proj.weight_shards[0].shape)}")
    print(f"max_abs_error={max_error:.3e}")
    print(f"outputs_close={close}")


def copy_dense_attention_to_tp_attention(dense: CausalSelfAttention, tp_attention: CausalSelfAttention) -> None:
    if not isinstance(dense.qkv, nn.Linear) or not isinstance(dense.proj, nn.Linear):
        raise TypeError("dense attention must use nn.Linear projections")
    if not isinstance(tp_attention.proj, RowParallelLinear):
        raise TypeError("TP attention must use RowParallelLinear output projection")

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


if __name__ == "__main__":
    main()
