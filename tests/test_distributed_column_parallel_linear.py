import os
import socket

import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedColumnParallelLinear,
    init_distributed_from_env,
    is_distributed_available,
)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed column-parallel linear tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_column_parallel_linear_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_column_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_column_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_gathered_output_matches_dense()
        _assert_local_output_matches_dense_slice()
        _assert_bias_false_matches_dense()
        _assert_copy_from_dense_slices_rank_local_parameters()
        _assert_gathered_input_gradients_match_dense()
        _assert_invalid_out_features_raise()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_gathered_output_matches_dense() -> None:
    torch.manual_seed(801)
    dense = nn.Linear(4, 8, bias=True)
    layer = DistributedColumnParallelLinear(4, 8, bias=True, gather_output=True)
    layer.copy_from_dense_(dense)
    x = torch.randn(3, 5, 4)

    torch.testing.assert_close(layer(x), dense(x), atol=1e-6, rtol=1e-6)


def _assert_local_output_matches_dense_slice() -> None:
    torch.manual_seed(802)
    dense = nn.Linear(4, 8, bias=True)
    layer = DistributedColumnParallelLinear(4, 8, bias=True, gather_output=False)
    layer.copy_from_dense_(dense)
    x = torch.randn(3, 5, 4)

    expected = dense(x)[..., layer.local_out_start : layer.local_out_end]
    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-6)


def _assert_bias_false_matches_dense() -> None:
    torch.manual_seed(803)
    dense = nn.Linear(4, 8, bias=False)
    layer = DistributedColumnParallelLinear(4, 8, bias=False, gather_output=True)
    layer.copy_from_dense_(dense)
    x = torch.randn(3, 5, 4)

    assert layer.bias is None
    torch.testing.assert_close(layer(x), dense(x), atol=1e-6, rtol=1e-6)


def _assert_copy_from_dense_slices_rank_local_parameters() -> None:
    torch.manual_seed(804)
    dense = nn.Linear(4, 8, bias=True)
    layer = DistributedColumnParallelLinear(4, 8, bias=True, gather_output=True)
    layer.copy_from_dense_(dense)

    start, end = layer.local_out_start, layer.local_out_end
    torch.testing.assert_close(layer.weight, dense.weight[start:end])
    assert layer.bias is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias, dense.bias[start:end])


def _assert_gathered_input_gradients_match_dense() -> None:
    torch.manual_seed(805)
    dense = nn.Linear(4, 8, bias=True)
    layer = DistributedColumnParallelLinear(4, 8, bias=True, gather_output=True)
    layer.copy_from_dense_(dense)

    dense_x = torch.randn(3, 5, 4, requires_grad=True)
    distributed_x = dense_x.detach().clone().requires_grad_()

    dense_loss = dense(dense_x).square().mean()
    distributed_loss = layer(distributed_x).square().mean()
    dense_loss.backward()
    distributed_loss.backward()

    torch.testing.assert_close(distributed_x.grad, dense_x.grad, atol=1e-6, rtol=1e-6)
    start, end = layer.local_out_start, layer.local_out_end
    torch.testing.assert_close(layer.weight.grad, dense.weight.grad[start:end], atol=1e-6, rtol=1e-6)
    assert layer.bias is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias.grad, dense.bias.grad[start:end], atol=1e-6, rtol=1e-6)


def _assert_invalid_out_features_raise() -> None:
    with pytest.raises(ValueError, match="out_features=7 must be divisible"):
        DistributedColumnParallelLinear(4, 7)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
