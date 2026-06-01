import os
import socket

import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedColumnParallelLinear,
    DistributedRowParallelLinear,
    init_distributed_from_env,
    is_distributed_available,
)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed MLP composition tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_mlp_composition_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_mlp_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_mlp_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_composed_mlp_matches_dense(world_size)
        _assert_invalid_intermediate_size_raises()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_composed_mlp_matches_dense(world_size: int) -> None:
    torch.manual_seed(9302)
    hidden_size = 4
    intermediate_size = 8
    dense_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
    dense_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
    activation = nn.GELU()

    dist_fc1 = DistributedColumnParallelLinear(
        hidden_size,
        intermediate_size,
        bias=True,
        gather_output=False,
    )
    dist_fc2 = DistributedRowParallelLinear(
        intermediate_size,
        hidden_size,
        bias=True,
        input_is_parallel=True,
    )
    dist_fc1.copy_from_dense_(dense_fc1)
    dist_fc2.copy_from_dense_(dense_fc2)

    dense_x = torch.randn(2, 3, hidden_size, requires_grad=True)
    dist_x = dense_x.detach().clone().requires_grad_()

    dense_y = dense_fc2(activation(dense_fc1(dense_x)))
    local_intermediate = dist_fc1(dist_x)
    dist_y = dist_fc2(activation(local_intermediate))

    torch.testing.assert_close(dist_y, dense_y, atol=1e-6, rtol=1e-6)
    assert dist_y.shape == (2, 3, hidden_size)

    dense_y.square().mean().backward()
    dist_y.square().mean().backward()

    assert dense_x.grad is not None
    assert dist_x.grad is not None
    torch.testing.assert_close(dist_x.grad, dense_x.grad, atol=1e-6, rtol=1e-6)

    start, end = dist_fc1.local_out_start, dist_fc1.local_out_end
    torch.testing.assert_close(dist_fc1.weight.grad, dense_fc1.weight.grad[start:end], atol=1e-6, rtol=1e-6)
    assert dist_fc1.bias is not None
    assert dense_fc1.bias is not None
    torch.testing.assert_close(dist_fc1.bias.grad, dense_fc1.bias.grad[start:end], atol=1e-6, rtol=1e-6)

    start, end = dist_fc2.local_in_start, dist_fc2.local_in_end
    torch.testing.assert_close(dist_fc2.weight.grad, dense_fc2.weight.grad[:, start:end], atol=1e-6, rtol=1e-6)
    assert dist_fc2.bias is not None
    assert dense_fc2.bias is not None
    torch.testing.assert_close(dist_fc2.bias.grad, dense_fc2.bias.grad, atol=1e-6, rtol=1e-6)


def _assert_invalid_intermediate_size_raises() -> None:
    with pytest.raises(ValueError, match="out_features=7 must be divisible"):
        DistributedColumnParallelLinear(4, 7, bias=True, gather_output=False)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
