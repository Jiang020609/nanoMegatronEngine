from collections.abc import Sequence

import torch
from torch import nn

from nano_megatron_engine.parallel import ColumnParallelLinear, FakeShardListCollectives, RowParallelLinear


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


def test_column_parallel_linear_default_collectives_match_linear():
    torch.manual_seed(701)
    linear = nn.Linear(8, 12)
    parallel = ColumnParallelLinear.from_linear(linear, tp_size=3)
    x = torch.randn(2, 4, 8)

    assert torch.allclose(parallel(x), linear(x), atol=1e-6, rtol=1e-6)


def test_row_parallel_linear_default_collectives_match_linear():
    torch.manual_seed(702)
    linear = nn.Linear(12, 8)
    parallel = RowParallelLinear.from_linear(linear, tp_size=3)
    x = torch.randn(2, 4, 12)

    assert torch.allclose(parallel(x), linear(x), atol=1e-6, rtol=1e-6)


def test_column_parallel_linear_uses_injected_all_gather_and_preserves_gradients():
    torch.manual_seed(703)
    collectives = RecordingShardListCollectives()
    linear = nn.Linear(8, 12)
    parallel = ColumnParallelLinear.from_linear(linear, tp_size=3, collectives=collectives)
    x = torch.randn(2, 4, 8, requires_grad=True)

    output = parallel(x)
    loss = output.square().mean()
    loss.backward()

    assert collectives.all_gather_calls == 1
    assert collectives.all_reduce_sum_calls == 0
    assert x.grad is not None
    assert all(shard.grad is not None for shard in parallel.weight_shards)
    if parallel.bias_shards is not None:
        assert all(shard.grad is not None for shard in parallel.bias_shards)


def test_row_parallel_linear_uses_injected_all_reduce_sum_and_preserves_gradients():
    torch.manual_seed(704)
    collectives = RecordingShardListCollectives()
    linear = nn.Linear(12, 8)
    parallel = RowParallelLinear.from_linear(linear, tp_size=3, collectives=collectives)
    x = torch.randn(2, 4, 12, requires_grad=True)

    output = parallel(x)
    loss = output.square().mean()
    loss.backward()

    assert collectives.all_reduce_sum_calls == 1
    assert collectives.all_gather_calls == 0
    assert x.grad is not None
    assert all(shard.grad is not None for shard in parallel.weight_shards)
    assert parallel.bias is not None
    assert parallel.bias.grad is not None


def test_column_parallel_linear_without_gather_does_not_call_all_gather():
    torch.manual_seed(705)
    collectives = RecordingShardListCollectives()
    linear = nn.Linear(8, 12)
    parallel = ColumnParallelLinear.from_linear(
        linear,
        tp_size=3,
        gather_output=False,
        collectives=collectives,
    )
    x = torch.randn(2, 4, 8)

    outputs = parallel(x)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert collectives.all_gather_calls == 0
