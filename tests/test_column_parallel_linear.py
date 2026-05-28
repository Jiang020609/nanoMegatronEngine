import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import ColumnParallelLinear


def test_column_parallel_linear_forward_and_backward_match_linear():
    torch.manual_seed(123)
    linear = nn.Linear(6, 10, bias=True)
    parallel = ColumnParallelLinear.from_linear(linear, tp_size=2)

    x = torch.randn(4, 3, 6, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    grad_output = torch.randn(4, 3, 10)

    ref_output = linear(x_ref)
    parallel_output = parallel(x)

    assert torch.allclose(parallel_output, ref_output, atol=1e-6, rtol=1e-6)

    ref_output.backward(grad_output)
    parallel_output.backward(grad_output)

    weight_grad = torch.cat([shard.grad for shard in parallel.weight_shards], dim=0)
    bias_grad = torch.cat([shard.grad for shard in parallel.bias_shards], dim=0)

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weight_grad, linear.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bias_grad, linear.bias.grad, atol=1e-6, rtol=1e-6)


def test_column_parallel_linear_merge_to_linear_round_trip():
    torch.manual_seed(124)
    linear = nn.Linear(8, 12, bias=True)
    parallel = ColumnParallelLinear.from_linear(linear, tp_size=3)

    merged = parallel.merge_to_linear()

    assert torch.allclose(merged.weight, linear.weight)
    assert torch.allclose(merged.bias, linear.bias)


def test_column_parallel_linear_can_return_local_outputs():
    torch.manual_seed(125)
    linear = nn.Linear(4, 8, bias=True)
    parallel = ColumnParallelLinear.from_linear(linear, tp_size=2, gather_output=False)
    x = torch.randn(2, 4)

    local_outputs = parallel(x)

    assert isinstance(local_outputs, tuple)
    assert len(local_outputs) == 2
    assert torch.allclose(torch.cat(local_outputs, dim=-1), linear(x), atol=1e-6, rtol=1e-6)


def test_column_parallel_linear_invalid_divisibility_raises():
    with pytest.raises(ValueError, match="out_features=10 must be divisible"):
        ColumnParallelLinear(4, 10, tp_size=3)

