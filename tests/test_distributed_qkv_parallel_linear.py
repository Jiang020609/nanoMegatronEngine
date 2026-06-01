import os
import socket

import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedQKVParallelLinear,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_qkv_parallel_linear_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        DistributedQKVParallelLinear(hidden_size=8, num_heads=4)


def test_distributed_qkv_parallel_linear_validates_hidden_head_relationship_before_distributed():
    with pytest.raises(ValueError, match="distributed QKV.*hidden_size=10.*num_heads=4"):
        DistributedQKVParallelLinear(hidden_size=10, num_heads=4)


def test_distributed_qkv_parallel_linear_validates_bias_type_before_distributed():
    with pytest.raises(TypeError, match="distributed QKV bias must be bool"):
        DistributedQKVParallelLinear(hidden_size=8, num_heads=4, bias=1)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed QKV parallel linear tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_qkv_parallel_linear_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_qkv_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_qkv_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_local_qkv_output_matches_dense_slices()
        _assert_bias_false_matches_dense_slices()
        _assert_copy_from_dense_slices_rank_local_parameters()
        _assert_input_gradients_match_dense_qkv()
        _assert_invalid_num_heads_divisibility_raises()
        _assert_invalid_forward_input_shape_raises()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_local_qkv_output_matches_dense_slices() -> None:
    torch.manual_seed(1001)
    hidden_size = 8
    num_heads = 4
    dense_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
    layer = DistributedQKVParallelLinear(hidden_size=hidden_size, num_heads=num_heads, bias=True)
    layer.copy_from_dense_(dense_qkv)
    x = torch.randn(2, 3, hidden_size)

    expected = _expected_local_qkv_output(layer, dense_qkv(x))
    actual = layer(x)

    assert actual.shape == (2, 3, 3 * layer.local_hidden)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _assert_bias_false_matches_dense_slices() -> None:
    torch.manual_seed(1002)
    hidden_size = 8
    num_heads = 4
    dense_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
    layer = DistributedQKVParallelLinear(hidden_size=hidden_size, num_heads=num_heads, bias=False)
    layer.copy_from_dense_(dense_qkv)
    x = torch.randn(2, 3, hidden_size)

    assert layer.bias is None
    expected = _expected_local_qkv_output(layer, dense_qkv(x))
    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-6)


def _assert_copy_from_dense_slices_rank_local_parameters() -> None:
    hidden_size = 8
    num_heads = 4
    dense_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
    with torch.no_grad():
        dense_qkv.weight.copy_(
            torch.arange(3 * hidden_size * hidden_size, dtype=torch.float32).view(3 * hidden_size, hidden_size)
        )
        dense_qkv.bias.copy_(torch.arange(3 * hidden_size, dtype=torch.float32))

    layer = DistributedQKVParallelLinear(hidden_size=hidden_size, num_heads=num_heads, bias=True)
    layer.copy_from_dense_(dense_qkv)

    q_start, q_end, k_start, k_end, v_start, v_end = _qkv_ranges(layer)
    expected_weight = torch.cat(
        [
            dense_qkv.weight[q_start:q_end],
            dense_qkv.weight[k_start:k_end],
            dense_qkv.weight[v_start:v_end],
        ],
        dim=0,
    )
    expected_bias = torch.cat(
        [
            dense_qkv.bias[q_start:q_end],
            dense_qkv.bias[k_start:k_end],
            dense_qkv.bias[v_start:v_end],
        ],
        dim=0,
    )

    torch.testing.assert_close(layer.weight, expected_weight)
    assert layer.bias is not None
    torch.testing.assert_close(layer.bias, expected_bias)


def _assert_input_gradients_match_dense_qkv() -> None:
    torch.manual_seed(1003)
    hidden_size = 8
    num_heads = 4
    dense_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
    layer = DistributedQKVParallelLinear(hidden_size=hidden_size, num_heads=num_heads, bias=True)
    layer.copy_from_dense_(dense_qkv)

    dense_x = torch.randn(2, 3, hidden_size, requires_grad=True)
    distributed_x = dense_x.detach().clone().requires_grad_()

    dense_loss = dense_qkv(dense_x).square().sum()
    distributed_loss = layer(distributed_x).square().sum()
    dense_loss.backward()
    distributed_loss.backward()

    assert dense_x.grad is not None
    assert distributed_x.grad is not None
    torch.testing.assert_close(distributed_x.grad, dense_x.grad, atol=1e-6, rtol=1e-6)

    q_start, q_end, k_start, k_end, v_start, v_end = _qkv_ranges(layer)
    assert layer.weight.grad is not None
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert dense_qkv.weight.grad is not None
    assert dense_qkv.bias is not None
    assert dense_qkv.bias.grad is not None
    expected_weight_grad = torch.cat(
        [
            dense_qkv.weight.grad[q_start:q_end],
            dense_qkv.weight.grad[k_start:k_end],
            dense_qkv.weight.grad[v_start:v_end],
        ],
        dim=0,
    )
    expected_bias_grad = torch.cat(
        [
            dense_qkv.bias.grad[q_start:q_end],
            dense_qkv.bias.grad[k_start:k_end],
            dense_qkv.bias.grad[v_start:v_end],
        ],
        dim=0,
    )
    torch.testing.assert_close(layer.weight.grad, expected_weight_grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(layer.bias.grad, expected_bias_grad, atol=1e-6, rtol=1e-6)


def _assert_invalid_num_heads_divisibility_raises() -> None:
    with pytest.raises(ValueError, match="distributed QKV.*strict head divisibility.*num_heads=3.*world_size=2"):
        DistributedQKVParallelLinear(hidden_size=6, num_heads=3)


def _assert_invalid_forward_input_shape_raises() -> None:
    layer = DistributedQKVParallelLinear(hidden_size=8, num_heads=4)
    with pytest.raises(ValueError, match="distributed QKV expected input last dimension hidden_size=8"):
        layer(torch.randn(2, 3, 7))


def _expected_local_qkv_output(layer: DistributedQKVParallelLinear, dense_qkv_output: torch.Tensor) -> torch.Tensor:
    q_start, q_end, k_start, k_end, v_start, v_end = _qkv_ranges(layer)
    return torch.cat(
        [
            dense_qkv_output[..., q_start:q_end],
            dense_qkv_output[..., k_start:k_end],
            dense_qkv_output[..., v_start:v_end],
        ],
        dim=-1,
    )


def _qkv_ranges(layer: DistributedQKVParallelLinear) -> tuple[int, int, int, int, int, int]:
    q_start = layer.local_start
    q_end = layer.local_end
    k_start = layer.hidden_size + layer.local_start
    k_end = layer.hidden_size + layer.local_end
    v_start = 2 * layer.hidden_size + layer.local_start
    v_end = 2 * layer.hidden_size + layer.local_end
    return q_start, q_end, k_start, k_end, v_start, v_end


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
