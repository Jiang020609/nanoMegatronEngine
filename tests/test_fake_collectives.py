import pytest
import torch

from nano_megatron_engine.parallel.fake_tp import (
    fake_all_gather,
    fake_all_reduce_sum,
    fake_reduce_scatter_sum,
    partition_range,
)


def test_fake_all_reduce_sum_correctness():
    shards = [torch.ones(2, 3), torch.full((2, 3), 2.0), torch.full((2, 3), 3.0)]

    reduced = fake_all_reduce_sum(shards)

    assert torch.equal(reduced, torch.full((2, 3), 6.0))


def test_fake_all_reduce_sum_preserves_gradients():
    shard_a = torch.ones(2, 3, requires_grad=True)
    shard_b = torch.full((2, 3), 2.0, requires_grad=True)

    loss = fake_all_reduce_sum([shard_a, shard_b]).square().sum()
    loss.backward()

    expected_grad = torch.full((2, 3), 6.0)
    assert torch.equal(shard_a.grad, expected_grad)
    assert torch.equal(shard_b.grad, expected_grad)


def test_fake_all_reduce_sum_errors():
    with pytest.raises(ValueError, match="shards must not be empty"):
        fake_all_reduce_sum([])

    with pytest.raises(ValueError, match="all-reduce shards must have matching shapes"):
        fake_all_reduce_sum([torch.ones(2, 3), torch.ones(2, 4)])


def test_fake_all_gather_correctness_and_uneven_sizes():
    shards = [torch.ones(2, 2), torch.full((2, 3), 2.0), torch.full((2, 1), 3.0)]

    gathered = fake_all_gather(shards, dim=-1)

    assert gathered.shape == (2, 6)
    assert torch.equal(gathered[:, :2], torch.ones(2, 2))
    assert torch.equal(gathered[:, 2:5], torch.full((2, 3), 2.0))
    assert torch.equal(gathered[:, 5:], torch.full((2, 1), 3.0))


def test_fake_all_gather_preserves_gradients():
    shard_a = torch.ones(2, 2, requires_grad=True)
    shard_b = torch.full((2, 3), 2.0, requires_grad=True)

    loss = fake_all_gather([shard_a, shard_b], dim=-1).sum()
    loss.backward()

    assert torch.equal(shard_a.grad, torch.ones_like(shard_a))
    assert torch.equal(shard_b.grad, torch.ones_like(shard_b))


def test_fake_all_gather_errors():
    with pytest.raises(ValueError, match="shards must not be empty"):
        fake_all_gather([])

    with pytest.raises(ValueError, match="same rank"):
        fake_all_gather([torch.ones(2, 3), torch.ones(2, 3, 1)])

    with pytest.raises(ValueError, match="non-gather dimensions"):
        fake_all_gather([torch.ones(2, 3), torch.ones(4, 3)], dim=-1)


def test_fake_reduce_scatter_sum_correctness_and_uneven_partitions():
    shard_a = torch.arange(10, dtype=torch.float32).view(2, 5)
    shard_b = torch.ones(2, 5)

    part0 = fake_reduce_scatter_sum([shard_a, shard_b], num_partitions=2, partition_idx=0, dim=-1)
    part1 = fake_reduce_scatter_sum([shard_a, shard_b], num_partitions=2, partition_idx=1, dim=-1)

    reduced = shard_a + shard_b
    assert part0.shape == (2, 3)
    assert part1.shape == (2, 2)
    assert torch.equal(part0, reduced[:, :3])
    assert torch.equal(part1, reduced[:, 3:])


def test_fake_reduce_scatter_sum_preserves_gradients():
    shard_a = torch.ones(2, 5, requires_grad=True)
    shard_b = torch.full((2, 5), 2.0, requires_grad=True)

    fake_reduce_scatter_sum([shard_a, shard_b], num_partitions=2, partition_idx=1, dim=-1).sum().backward()

    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
    assert torch.equal(shard_a.grad, expected)
    assert torch.equal(shard_b.grad, expected)


def test_partition_range_covers_indices_once():
    ranges = [partition_range(7, 3, idx) for idx in range(3)]
    covered = [item for start, end in ranges for item in range(start, end)]

    assert ranges == [(0, 3), (3, 5), (5, 7)]
    assert covered == list(range(7))
