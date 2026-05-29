import torch
from torch import nn

from nano_megatron_engine.model import GPTConfig, GPTModel
from nano_megatron_engine.parallel import VocabParallelEmbedding, VocabParallelLMHead
from nano_megatron_engine.parallel.fake_tp import partition_range
from tests.test_tp_gpt_forward import copy_dense_gpt_to_tp_gpt


def test_vocab_parallel_embedding_matches_dense_for_divisible_vocab():
    torch.manual_seed(550)
    dense = nn.Embedding(64, 16)
    parallel = VocabParallelEmbedding.from_embedding(dense, tp_size=2)
    input_ids = torch.tensor([[0, 1, 31, 32], [33, 44, 62, 63]])

    output = parallel(input_ids)

    assert output.shape == (2, 4, 16)
    torch.testing.assert_close(output, dense(input_ids), atol=1e-6, rtol=1e-5)


def test_vocab_parallel_embedding_matches_dense_for_uneven_vocab():
    torch.manual_seed(551)
    dense = nn.Embedding(65, 16)
    parallel = VocabParallelEmbedding.from_embedding(dense, tp_size=2)
    input_ids = torch.tensor([[0, 1, 32, 33], [34, 48, 63, 64]])

    output = parallel(input_ids)

    assert parallel.vocab_ranges == [(0, 33), (33, 65)]
    assert output.shape == (2, 4, 16)
    torch.testing.assert_close(output, dense(input_ids), atol=1e-6, rtol=1e-5)


def test_vocab_parallel_lm_head_matches_dense_for_uneven_vocab_with_bias():
    torch.manual_seed(552)
    dense = nn.Linear(16, 65, bias=True)
    parallel = VocabParallelLMHead.from_linear(dense, tp_size=2)
    x = torch.randn(2, 4, 16)

    logits = parallel(x)

    assert parallel.vocab_ranges == [(0, 33), (33, 65)]
    assert logits.shape == (2, 4, 65)
    torch.testing.assert_close(logits, dense(x), atol=1e-6, rtol=1e-5)


def test_fake_tp_gpt_preserves_embedding_lm_head_weight_tying():
    config = GPTConfig(vocab_size=65, block_size=8, n_layer=1, n_head=2, n_embd=16, tensor_parallel_size=2)
    model = GPTModel(config)

    assert isinstance(model.token_embedding, VocabParallelEmbedding)
    assert isinstance(model.lm_head, VocabParallelLMHead)
    for embedding_weight, lm_head_weight in zip(model.token_embedding.weight_shards, model.lm_head.weight_shards):
        assert embedding_weight is lm_head_weight


def test_dense_and_fake_tp_gpt_logits_match_for_divisible_and_uneven_vocab():
    for vocab_size in (64, 65):
        torch.manual_seed(553 + vocab_size)
        dense_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            tensor_parallel_size=1,
        )
        tp_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            tensor_parallel_size=2,
        )
        dense = GPTModel(dense_config)
        tp_model = GPTModel(tp_config)
        copy_dense_gpt_to_tp_gpt(dense, tp_model)
        dense.eval()
        tp_model.eval()
        input_ids = torch.randint(0, vocab_size, (2, dense_config.block_size))

        dense_logits, _ = dense(input_ids, targets=input_ids)
        tp_logits, _ = tp_model(input_ids, targets=input_ids)

        assert tp_logits.shape == (2, dense_config.block_size, vocab_size)
        torch.testing.assert_close(tp_logits, dense_logits, atol=1e-6, rtol=1e-5)


def test_partition_range_example_for_vocab_parallelism():
    assert partition_range(65, 2, 0) == (0, 33)
    assert partition_range(65, 2, 1) == (33, 65)
