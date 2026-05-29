import os
import socket

import pytest
import torch

from nano_megatron_engine.parallel import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_rank,
    get_world_size,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_public_api_imports():
    assert callable(init_distributed_from_env)
    assert callable(distributed_all_reduce_sum)
    assert callable(distributed_all_gather)
    assert callable(distributed_reduce_scatter_sum)


def test_distributed_wrappers_raise_cleanly_when_not_initialized():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    message = "not available|not initialized"
    with pytest.raises(RuntimeError, match=message):
        distributed_all_reduce_sum(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        distributed_all_gather(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        distributed_reduce_scatter_sum(torch.ones(2))
    with pytest.raises(RuntimeError, match=message):
        get_rank()
    with pytest.raises(RuntimeError, match=message):
        get_world_size()


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed smoke tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_collectives_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        init_distributed_from_env("gloo")
        assert get_rank() == rank
        assert get_world_size() == world_size

        local = torch.full((2, 3), float(rank + 1))
        reduced = distributed_all_reduce_sum(local)
        assert torch.equal(reduced, torch.full((2, 3), 3.0))

        gather_input = torch.full((2, rank + 2), float(rank + 1))
        gathered = distributed_all_gather(gather_input, dim=-1)
        assert gathered.shape == (2, 5)
        assert torch.equal(gathered[:, :2], torch.ones(2, 2))
        assert torch.equal(gathered[:, 2:], torch.full((2, 3), 2.0))

        base = torch.arange(10, dtype=torch.float32).view(2, 5)
        scattered = distributed_reduce_scatter_sum(base + rank, dim=-1)
        expected_reduced = base * 2 + 1
        expected = expected_reduced[:, :3] if rank == 0 else expected_reduced[:, 3:]
        assert torch.equal(scattered, expected)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
