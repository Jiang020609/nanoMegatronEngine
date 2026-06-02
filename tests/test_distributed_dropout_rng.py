import os
import socket

import pytest
import torch

from nano_megatron_engine.model.distributed_transformer_block import DistributedTransformerBlock
from nano_megatron_engine.parallel import (
    RNGStateTracker,
    TrackedDropout,
    init_distributed_from_env,
    is_distributed_available,
)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed dropout RNG tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_dropout_rng_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_dropout_rng_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_dropout_rng_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_rank_local_and_replicated_streams(rank, world_size)
        _assert_distributed_block_tracked_dropout_replays(rank, world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_rank_local_and_replicated_streams(rank: int, world_size: int) -> None:
    import torch.distributed as dist

    tracker = RNGStateTracker(device="cpu")
    tracker.add("attention_dropout", 1801 + rank)
    tracker.add("residual_dropout", 1802)
    x = torch.ones(10, 10)
    rank_local_dropout = TrackedDropout(0.4, rng_tracker=tracker, rng_name="attention_dropout")
    replicated_dropout = TrackedDropout(0.4, rng_tracker=tracker, rng_name="residual_dropout")
    rank_local_dropout.train()
    replicated_dropout.train()

    rank_local_y = rank_local_dropout(x)
    replicated_y = replicated_dropout(x)

    rank_local_gathered = [torch.zeros_like(rank_local_y) for _ in range(world_size)]
    replicated_gathered = [torch.zeros_like(replicated_y) for _ in range(world_size)]
    dist.all_gather(rank_local_gathered, rank_local_y)
    dist.all_gather(replicated_gathered, replicated_y)

    if rank == 0:
        assert not torch.equal(rank_local_gathered[0], rank_local_gathered[1])
        torch.testing.assert_close(replicated_gathered[0], replicated_gathered[1])


def _assert_distributed_block_tracked_dropout_replays(rank: int, world_size: int) -> None:
    import torch.distributed as dist

    torch.manual_seed(1803)
    tracker = RNGStateTracker(device="cpu")
    tracker.add("attention_dropout", 1804 + rank)
    tracker.add("residual_dropout", 1805)
    block = DistributedTransformerBlock(
        hidden_size=8,
        num_heads=4,
        block_size=8,
        mlp_hidden_size=32,
        dropout=0.25,
        rng_tracker=tracker,
    )
    block.train()
    x = torch.randn(2, 5, 8)
    initial_attention_state = tracker.get_state("attention_dropout")
    initial_residual_state = tracker.get_state("residual_dropout")

    first = block(x)
    advanced_attention_state = tracker.get_state("attention_dropout")
    advanced_residual_state = tracker.get_state("residual_dropout")
    tracker.set_state("attention_dropout", initial_attention_state)
    tracker.set_state("residual_dropout", initial_residual_state)
    replay = block(x)

    torch.testing.assert_close(replay, first)
    assert not torch.equal(advanced_attention_state.cpu, initial_attention_state.cpu)
    assert not torch.equal(advanced_residual_state.cpu, initial_residual_state.cpu)

    gathered = [torch.zeros_like(first) for _ in range(world_size)]
    dist.all_gather(gathered, first)
    if rank == 0:
        torch.testing.assert_close(gathered[0], gathered[1])


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
