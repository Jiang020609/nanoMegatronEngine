import pytest
import torch

from nano_megatron_engine.parallel import (
    DistributedRankLocalCollectives,
    FakeShardListCollectives,
    RankLocalCollectiveProtocol,
    ShardListCollectiveProtocol,
    is_distributed_initialized,
)


def test_collective_adapters_are_public_api():
    assert FakeShardListCollectives is not None
    assert DistributedRankLocalCollectives is not None
    assert ShardListCollectiveProtocol is not RankLocalCollectiveProtocol


def test_fake_shard_list_adapter_all_reduce_sum_and_gradients():
    adapter = FakeShardListCollectives()
    shard_a = torch.ones(2, 3, requires_grad=True)
    shard_b = torch.full((2, 3), 2.0, requires_grad=True)

    reduced = adapter.all_reduce_sum([shard_a, shard_b])

    assert torch.equal(reduced, torch.full((2, 3), 3.0))
    reduced.sum().backward()
    assert torch.equal(shard_a.grad, torch.ones_like(shard_a))
    assert torch.equal(shard_b.grad, torch.ones_like(shard_b))


def test_fake_shard_list_adapter_all_gather_and_gradients():
    adapter = FakeShardListCollectives()
    shard_a = torch.arange(4, dtype=torch.float32).view(2, 2).requires_grad_()
    shard_b = torch.ones(2, 3, requires_grad=True)

    gathered = adapter.all_gather([shard_a, shard_b], dim=-1)

    assert gathered.shape == (2, 5)
    assert torch.equal(gathered[:, :2], shard_a)
    assert torch.equal(gathered[:, 2:], shard_b)
    gathered.sum().backward()
    assert torch.equal(shard_a.grad, torch.ones_like(shard_a))
    assert torch.equal(shard_b.grad, torch.ones_like(shard_b))


def test_fake_shard_list_adapter_reduce_scatter_sum_and_partition_range():
    adapter = FakeShardListCollectives()
    shard_a = torch.arange(10, dtype=torch.float32).view(2, 5).requires_grad_()
    shard_b = torch.ones(2, 5, requires_grad=True)

    scattered = adapter.reduce_scatter_sum(
        [shard_a, shard_b],
        num_partitions=2,
        partition_idx=0,
        dim=-1,
    )

    assert adapter.partition_range(5, 2, 0) == (0, 3)
    assert scattered.shape == (2, 3)
    assert torch.equal(scattered, (shard_a + shard_b)[:, :3])

    scattered.sum().backward()
    expected_grad = torch.zeros_like(shard_a)
    expected_grad[:, :3] = 1
    assert torch.equal(shard_a.grad, expected_grad)
    assert torch.equal(shard_b.grad, expected_grad)


def test_distributed_rank_local_adapter_fails_clearly_when_uninitialized():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    adapter = DistributedRankLocalCollectives()
    message = "torch.distributed.*(not available|not initialized).*init_distributed_from_env"

    assert adapter.is_initialized() is False
    with pytest.raises(RuntimeError, match=message):
        adapter.all_reduce_sum(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        adapter.all_gather(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        adapter.reduce_scatter_sum(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        adapter.get_rank()
    with pytest.raises(RuntimeError, match=message):
        adapter.get_world_size()
