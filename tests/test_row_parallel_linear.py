import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import RowParallelLinear


def test_row_parallel_linear_forward_and_backward_match_linear():
    torch.manual_seed(223)
    linear = nn.Linear(8, 5, bias=True)
    parallel = RowParallelLinear.from_linear(linear, tp_size=4)

    x = torch.randn(3, 2, 8, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    grad_output = torch.randn(3, 2, 5)

    ref_output = linear(x_ref)
    parallel_output = parallel(x)

    assert torch.allclose(parallel_output, ref_output, atol=1e-6, rtol=1e-6)

    ref_output.backward(grad_output)
    parallel_output.backward(grad_output)

    weight_grad = torch.cat([shard.grad for shard in parallel.weight_shards], dim=1)

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weight_grad, linear.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(parallel.bias.grad, linear.bias.grad, atol=1e-6, rtol=1e-6)


def test_row_parallel_linear_merge_to_linear_round_trip():
    torch.manual_seed(224)
    linear = nn.Linear(12, 7, bias=True)
    parallel = RowParallelLinear.from_linear(linear, tp_size=3)

    merged = parallel.merge_to_linear()

    assert torch.allclose(merged.weight, linear.weight)
    assert torch.allclose(merged.bias, linear.bias)


def test_row_parallel_linear_invalid_divisibility_raises():
    with pytest.raises(ValueError, match="in_features=10 must be divisible"):
        RowParallelLinear(10, 4, tp_size=3)

