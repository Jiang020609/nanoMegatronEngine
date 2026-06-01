import os
import socket
import warnings

import pytest
import torch

from nano_megatron_engine.parallel import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_backend,
    get_expected_device_type,
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
    assert callable(get_backend)
    assert callable(get_expected_device_type)


def test_distributed_wrappers_raise_cleanly_when_not_initialized():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    message = "torch.distributed.*(not available|not initialized).*init_distributed_from_env"
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


def test_distributed_wrappers_validate_inputs_before_collectives():
    with pytest.raises(TypeError, match="torch.Tensor"):
        distributed_all_reduce_sum([torch.ones(2)])
    with pytest.raises(TypeError, match="torch.Tensor"):
        distributed_all_gather(object())
    with pytest.raises(TypeError, match="torch.Tensor"):
        distributed_reduce_scatter_sum("not a tensor")

    with pytest.raises(ValueError, match="dim=3"):
        distributed_all_gather(torch.ones(2), dim=3)
    with pytest.raises(ValueError, match="dim=3"):
        distributed_reduce_scatter_sum(torch.ones(2), dim=3)

    meta_tensor = torch.empty(2, device="meta")
    with pytest.raises(ValueError, match="expects cpu tensors"):
        distributed_all_reduce_sum(meta_tensor)

    sparse_tensor = torch.eye(2).to_sparse()
    with pytest.raises(ValueError, match="dense strided"):
        distributed_all_gather(sparse_tensor)


def test_init_distributed_from_env_validates_contract(monkeypatch):
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    with pytest.raises(ValueError, match="backend='gloo'.*backend='nccl'"):
        init_distributed_from_env("mpi")
    with pytest.raises(ValueError, match="timeout_seconds"):
        init_distributed_from_env(timeout_seconds=0)

    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env.*RANK"):
        init_distributed_from_env("gloo")

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29599")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "2")
    with pytest.raises(ValueError, match="RANK=2"):
        init_distributed_from_env("gloo")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "")
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        init_distributed_from_env("gloo")

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "70000")
    with pytest.raises(ValueError, match="MASTER_PORT"):
        init_distributed_from_env("gloo")


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


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed smoke tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_world_size_one_collectives_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    port = _find_free_port()
    mp.spawn(_world_size_one_worker, args=(port,), nprocs=1, join=True)


def _distributed_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        assert get_rank() == rank
        assert get_world_size() == world_size
        assert get_backend() == "gloo"
        assert get_expected_device_type() == "cpu"

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

        bad_reduce = torch.ones(2, 3) if rank == 0 else torch.ones(2, 4)
        with pytest.raises(ValueError, match="matching tensor shapes"):
            distributed_all_reduce_sum(bad_reduce)

        bad_gather = torch.ones(2, 2) if rank == 0 else torch.ones(3, 3)
        with pytest.raises(ValueError, match="non-gather dimensions"):
            distributed_all_gather(bad_gather, dim=-1)

        bad_dtype = torch.ones(2, 2) if rank == 0 else torch.ones(2, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="matching tensor dtypes"):
            distributed_all_reduce_sum(bad_dtype)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*c10d::allreduce_.*", category=UserWarning)
            autograd_input = torch.full((2, 3), float(rank + 1), requires_grad=True)
            autograd_output = distributed_all_reduce_sum(autograd_input)
            autograd_output.square().sum().backward()
        assert autograd_input.grad is not None
        assert torch.equal(autograd_input.grad, torch.full((2, 3), 6.0))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _world_size_one_worker(rank: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "1"
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        assert get_rank() == 0
        assert get_world_size() == 1
        assert get_backend() == "gloo"
        assert get_expected_device_type() == "cpu"

        local = torch.arange(6, dtype=torch.float32).view(2, 3)
        assert torch.equal(distributed_all_reduce_sum(local), local)
        assert torch.equal(distributed_all_gather(local, dim=-1), local)
        assert torch.equal(distributed_reduce_scatter_sum(local, dim=-1), local)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
