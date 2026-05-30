from collections.abc import Sequence

import torch
from torch import nn

from nano_megatron_engine.parallel import (
    FakeShardListCollectives,
    VocabParallelEmbedding,
    VocabParallelLMHead,
)


class RecordingShardListCollectives:
    def __init__(self) -> None:
        self.delegate = FakeShardListCollectives()
        self.all_gather_calls = 0
        self.all_reduce_sum_calls = 0

    def all_reduce_sum(self, shards: Sequence[torch.Tensor]) -> torch.Tensor:
        self.all_reduce_sum_calls += 1
        return self.delegate.all_reduce_sum(shards)

    def all_gather(self, shards: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
        self.all_gather_calls += 1
        return self.delegate.all_gather(shards, dim=dim)

    def reduce_scatter_sum(
        self,
        shards: Sequence[torch.Tensor],
        num_partitions: int,
        partition_idx: int,
        dim: int = -1,
    ) -> torch.Tensor:
        return self.delegate.reduce_scatter_sum(
            shards,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
            dim=dim,
        )

    def partition_range(self, total_size: int, num_partitions: int, partition_idx: int) -> tuple[int, int]:
        return self.delegate.partition_range(total_size, num_partitions, partition_idx)


def test_vocab_parallel_embedding_default_collectives_match_embedding():
    torch.manual_seed(801)
    embedding = nn.Embedding(65, 16)
    parallel = VocabParallelEmbedding.from_embedding(embedding, tp_size=2)
    input_ids = torch.tensor([[0, 3, 32, 33, 64], [4, 20, 34, 48, 63]])

    assert torch.allclose(parallel(input_ids), embedding(input_ids), atol=1e-6, rtol=1e-6)


def test_vocab_parallel_lm_head_default_collectives_match_linear():
    torch.manual_seed(802)
    linear = nn.Linear(16, 65, bias=True)
    parallel = VocabParallelLMHead.from_linear(linear, tp_size=2)
    x = torch.randn(2, 4, 16)

    assert torch.allclose(parallel(x), linear(x), atol=1e-6, rtol=1e-6)


def test_vocab_parallel_embedding_uses_injected_all_reduce_sum_and_preserves_gradients():
    torch.manual_seed(803)
    collectives = RecordingShardListCollectives()
    embedding = nn.Embedding(65, 16)
    parallel = VocabParallelEmbedding.from_embedding(embedding, tp_size=2, collectives=collectives)
    input_ids = torch.tensor([[0, 3, 32, 33, 64], [4, 20, 34, 48, 63]])

    output = parallel(input_ids)
    output.square().mean().backward()

    assert collectives.all_reduce_sum_calls == 1
    assert collectives.all_gather_calls == 0
    assert all(weight.grad is not None for weight in parallel.weight_shards)


def test_vocab_parallel_lm_head_uses_injected_all_gather_and_preserves_gradients():
    torch.manual_seed(804)
    collectives = RecordingShardListCollectives()
    linear = nn.Linear(16, 65, bias=True)
    parallel = VocabParallelLMHead.from_linear(linear, tp_size=2, collectives=collectives)
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = parallel(x)
    output.square().mean().backward()

    assert collectives.all_gather_calls == 1
    assert collectives.all_reduce_sum_calls == 0
    assert x.grad is not None
    assert all(weight.grad is not None for weight in parallel.weight_shards)
    assert parallel.bias_shards is not None
    assert all(bias.grad is not None for bias in parallel.bias_shards)


def test_vocab_parallel_uneven_ranges_and_merge_round_trip_with_injected_collectives():
    torch.manual_seed(805)
    collectives = RecordingShardListCollectives()
    embedding = nn.Embedding(65, 8)
    lm_head = nn.Linear(8, 65, bias=True)

    parallel_embedding = VocabParallelEmbedding.from_embedding(embedding, tp_size=2, collectives=collectives)
    parallel_lm_head = VocabParallelLMHead.from_linear(lm_head, tp_size=2, collectives=collectives)

    assert parallel_embedding.vocab_ranges == [(0, 33), (33, 65)]
    assert parallel_lm_head.vocab_ranges == [(0, 33), (33, 65)]

    merged_embedding = parallel_embedding.merge_to_embedding()
    merged_lm_head = parallel_lm_head.merge_to_linear()

    assert torch.allclose(merged_embedding.weight, embedding.weight)
    assert torch.allclose(merged_lm_head.weight, lm_head.weight)
    assert merged_lm_head.bias is not None
    assert lm_head.bias is not None
    assert torch.allclose(merged_lm_head.bias, lm_head.bias)
