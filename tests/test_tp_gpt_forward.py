import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig, GPTModel
from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.model.mlp import MLP
from nano_megatron_engine.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLMHead,
)
from nano_megatron_engine.parallel.fake_tp import split_tensor_along_dim


def test_gpt_forward_with_and_without_fake_tp_mlp():
    input_ids = torch.randint(0, 32, (2, 8))

    for tensor_parallel_size in (1, 2):
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            tensor_parallel_size=tensor_parallel_size,
        )
        model = GPTModel(config)

        logits, loss = model(input_ids, targets=input_ids)

        assert logits.shape == (2, 8, 32)
        assert loss is not None
        assert loss.ndim == 0


def test_gpt_fake_tp_mlp_backward_smoke():
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))

    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_dense_gpt_logits_match_fake_tp_gpt_logits_after_weight_copy():
    torch.manual_seed(446)
    dense_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        dropout=0.0,
        tensor_parallel_size=1,
    )
    tp_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    dense = GPTModel(dense_config)
    tp_model = GPTModel(tp_config)
    copy_dense_gpt_to_tp_gpt(dense, tp_model)
    dense.eval()
    tp_model.eval()
    input_ids = torch.randint(0, dense_config.vocab_size, (2, dense_config.block_size))

    dense_logits, dense_loss = dense(input_ids, targets=input_ids)
    tp_logits, tp_loss = tp_model(input_ids, targets=input_ids)

    torch.testing.assert_close(tp_logits, dense_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(tp_loss, dense_loss, atol=1e-6, rtol=1e-5)


def test_no_bias_dense_gpt_logits_match_fake_tp_gpt_logits_after_weight_copy():
    torch.manual_seed(447)
    dense_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        bias=False,
        dropout=0.0,
        tensor_parallel_size=1,
    )
    tp_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        bias=False,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    dense = GPTModel(dense_config)
    tp_model = GPTModel(tp_config)
    copy_dense_gpt_to_tp_gpt(dense, tp_model)
    dense.eval()
    tp_model.eval()
    input_ids = torch.randint(0, dense_config.vocab_size, (2, dense_config.block_size))

    dense_logits, dense_loss = dense(input_ids, targets=input_ids)
    tp_logits, tp_loss = tp_model(input_ids, targets=input_ids)

    torch.testing.assert_close(tp_logits, dense_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(tp_loss, dense_loss, atol=1e-6, rtol=1e-5)


def copy_dense_gpt_to_tp_gpt(dense: GPTModel, tp_model: GPTModel) -> None:
    with torch.no_grad():
        copy_dense_embedding_to_tp_embedding(dense.token_embedding, tp_model.token_embedding)
        tp_model.position_embedding.weight.copy_(dense.position_embedding.weight)
        tp_model.ln_f.weight.copy_(dense.ln_f.weight)
        tp_model.ln_f.bias.copy_(dense.ln_f.bias)

        for dense_block, tp_block in zip(dense.blocks, tp_model.blocks):
            tp_block.ln_1.weight.copy_(dense_block.ln_1.weight)
            tp_block.ln_1.bias.copy_(dense_block.ln_1.bias)
            tp_block.ln_2.weight.copy_(dense_block.ln_2.weight)
            tp_block.ln_2.bias.copy_(dense_block.ln_2.bias)
            copy_dense_attention_to_tp_attention(dense_block.attn, tp_block.attn)
            copy_dense_mlp_to_tp_mlp(dense_block.mlp, tp_block.mlp)


def copy_dense_embedding_to_tp_embedding(dense_embedding: nn.Embedding, tp_embedding: nn.Module) -> None:
    if isinstance(tp_embedding, nn.Embedding):
        tp_embedding.weight.copy_(dense_embedding.weight)
        return

    assert isinstance(tp_embedding, VocabParallelEmbedding)
    for shard, (start, end) in zip(tp_embedding.weight_shards, tp_embedding.vocab_ranges):
        shard.copy_(dense_embedding.weight[start:end])


def copy_dense_attention_to_tp_attention(dense: CausalSelfAttention, tp_attention: CausalSelfAttention) -> None:
    assert isinstance(dense.qkv, nn.Linear)
    assert isinstance(dense.proj, nn.Linear)
    assert isinstance(tp_attention.proj, RowParallelLinear)

    hidden_size = dense.n_embd
    local_hidden = tp_attention.local_hidden

    bias_shards = tp_attention.qkv_bias_shards
    for shard_idx, weight_shard in enumerate(tp_attention.qkv_weight_shards):
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
        if dense.qkv.bias is not None and bias_shards is not None:
            bias_shards[shard_idx].copy_(
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
    if dense.proj.bias is not None and tp_attention.proj.bias is not None:
        tp_attention.proj.bias.copy_(dense.proj.bias)


def copy_dense_mlp_to_tp_mlp(dense_mlp: MLP, tp_mlp: MLP) -> None:
    first_linear = dense_mlp.net[0]
    second_linear = dense_mlp.net[2]
    first_tp = tp_mlp.net[0]
    second_tp = tp_mlp.net[2]

    assert isinstance(first_linear, nn.Linear)
    assert isinstance(second_linear, nn.Linear)
    assert isinstance(first_tp, ColumnParallelLinear)
    assert isinstance(second_tp, RowParallelLinear)

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
