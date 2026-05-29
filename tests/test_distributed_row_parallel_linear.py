import os
import socket

import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedRowParallelLinear,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_row_parallel_linear_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        DistributedRowParallelLinear(6, 4)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed row-parallel linear tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_row_parallel_linear_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_row_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_row_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_full_input_output_matches_dense()
        _assert_parallel_input_output_matches_dense()
        _assert_bias_false_matches_dense()
        _assert_bias_is_applied_once()
        _assert_copy_from_dense_slices_rank_local_parameters()
        _assert_invalid_in_features_raise()
        _assert_invalid_input_shapes_raise()
        _assert_gradients_match_dense_slices()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_full_input_output_matches_dense() -> None:
    torch.manual_seed(901)
    dense = nn.Linear(6, 4, bias=True)
    layer = DistributedRowParallelLinear(6, 4, bias=True, input_is_parallel=False)
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 3, 6)

    torch.testing.assert_close(layer(x), dense(x), atol=1e-6, rtol=1e-6)


def _assert_parallel_input_output_matches_dense() -> None:
    torch.manual_seed(902)
    dense = nn.Linear(6, 4, bias=True)
    layer = DistributedRowParallelLinear(6, 4, bias=True, input_is_parallel=True)
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 3, 6)
    local_x = x[..., layer.local_in_start : layer.local_in_end].contiguous()

    torch.testing.assert_close(layer(local_x), dense(x), atol=1e-6, rtol=1e-6)


def _assert_bias_false_matches_dense() -> None:
    torch.manual_seed(903)
    dense = nn.Linear(6, 4, bias=False)
    layer = DistributedRowParallelLinear(6, 4, bias=False, input_is_parallel=False)
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 3, 6)

    assert layer.bias is None
    torch.testing.assert_close(layer(x), dense(x), atol=1e-6, rtol=1e-6)


def _assert_bias_is_applied_once() -> None:
    dense = nn.Linear(6, 4, bias=True)
    with torch.no_grad():
        dense.weight.zero_()
        dense.bias.copy_(torch.tensor([1.0, -2.0, 3.0, -4.0]))
    layer = DistributedRowParallelLinear(6, 4, bias=True, input_is_parallel=False)
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 3, 6)

    expected = dense.bias.view(1, 1, -1).expand(2, 3, 4)
    torch.testing.assert_close(layer(x), expected)


def _assert_copy_from_dense_slices_rank_local_parameters() -> None:
    torch.manual_seed(904)
    dense = nn.Linear(6, 4, bias=True)
    layer = DistributedRowParallelLinear(6, 4, bias=True, input_is_parallel=False)
    layer.copy_from_dense_(dense)

    start, end = layer.local_in_start, layer.local_in_end
    torch.testing.assert_close(layer.weight, dense.weight[:, start:end])
    assert layer.bias is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias, dense.bias)


def _assert_invalid_in_features_raise() -> None:
    with pytest.raises(ValueError, match="row-parallel in_features=5 must be divisible"):
        DistributedRowParallelLinear(5, 4)


def _assert_invalid_input_shapes_raise() -> None:
    full_input_layer = DistributedRowParallelLinear(6, 4, input_is_parallel=False)
    with pytest.raises(ValueError, match="input_is_parallel=False"):
        full_input_layer(torch.randn(2, 3, 3))

    parallel_input_layer = DistributedRowParallelLinear(6, 4, input_is_parallel=True)
    with pytest.raises(ValueError, match="input_is_parallel=True"):
        parallel_input_layer(torch.randn(2, 3, 6))


def _assert_gradients_match_dense_slices() -> None:
    torch.manual_seed(905)
    dense = nn.Linear(6, 4, bias=True)
    layer = DistributedRowParallelLinear(6, 4, bias=True, input_is_parallel=True)
    layer.copy_from_dense_(dense)

    dense_x = torch.randn(2, 3, 6, requires_grad=True)
    local_x = dense_x.detach()[..., layer.local_in_start : layer.local_in_end].contiguous().requires_grad_()

    dense_loss = dense(dense_x).square().mean()
    distributed_loss = layer(local_x).square().mean()
    dense_loss.backward()
    distributed_loss.backward()

    start, end = layer.local_in_start, layer.local_in_end
    torch.testing.assert_close(layer.weight.grad, dense.weight.grad[:, start:end], atol=1e-6, rtol=1e-6)
    assert layer.bias is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias.grad, dense.bias.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_x.grad, dense_x.grad[..., start:end], atol=1e-6, rtol=1e-6)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
