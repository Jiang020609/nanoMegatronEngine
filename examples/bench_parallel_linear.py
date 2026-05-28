"""Compare nn.Linear with fake tensor-parallel linear layers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from nano_megatron_engine.parallel import ColumnParallelLinear, RowParallelLinear


@dataclass(frozen=True)
class LinearComparison:
    name: str
    output_max_error: float
    input_grad_max_error: float
    reference_time_ms: float
    parallel_time_ms: float


def main() -> None:
    torch.manual_seed(2026)
    device = torch.device("cpu")
    shapes = [
        {"batch": 8, "seq": 4, "in_features": 16, "out_features": 32, "tp_size": 2},
        {"batch": 8, "seq": 4, "in_features": 32, "out_features": 16, "tp_size": 4},
    ]

    print("Fake tensor parallel linear benchmark")
    print("device: cpu")
    print("note: fake TP is educational single-process code and is not expected to be faster.")
    print()

    for shape in shapes:
        linear = nn.Linear(shape["in_features"], shape["out_features"], bias=True).to(device)
        x = torch.randn(shape["batch"], shape["seq"], shape["in_features"], device=device)
        column = ColumnParallelLinear.from_linear(linear, tp_size=shape["tp_size"])
        row = RowParallelLinear.from_linear(linear, tp_size=shape["tp_size"])

        print(
            f"shape: batch={shape['batch']} seq={shape['seq']} "
            f"in={shape['in_features']} out={shape['out_features']} tp={shape['tp_size']}"
        )
        for result in (
            compare_layers("ColumnParallelLinear", linear, column, x),
            compare_layers("RowParallelLinear", linear, row, x),
        ):
            print(
                f"  {result.name}: output_err={result.output_max_error:.3e} "
                f"input_grad_err={result.input_grad_max_error:.3e} "
                f"nn_time={result.reference_time_ms:.3f}ms "
                f"fake_tp_time={result.parallel_time_ms:.3f}ms"
            )
        print()


def compare_layers(name: str, reference: nn.Linear, parallel: nn.Module, x: torch.Tensor) -> LinearComparison:
    output_error, input_grad_error = max_errors(reference, parallel, x)
    reference_time = time_forward_backward(reference, x)
    parallel_time = time_forward_backward(parallel, x)
    return LinearComparison(
        name=name,
        output_max_error=output_error,
        input_grad_max_error=input_grad_error,
        reference_time_ms=reference_time,
        parallel_time_ms=parallel_time,
    )


def max_errors(reference: nn.Linear, parallel: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    reference.zero_grad(set_to_none=True)
    parallel.zero_grad(set_to_none=True)

    x_ref = x.detach().clone().requires_grad_()
    x_parallel = x.detach().clone().requires_grad_()
    reference_output = reference(x_ref)
    parallel_output = parallel(x_parallel)
    grad_output = torch.randn_like(reference_output)

    reference_output.backward(grad_output)
    parallel_output.backward(grad_output)

    output_error = (parallel_output.detach() - reference_output.detach()).abs().max().item()
    input_grad_error = (x_parallel.grad - x_ref.grad).abs().max().item()
    return output_error, input_grad_error


def time_forward_backward(layer: nn.Module, x: torch.Tensor, iterations: int = 30) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        layer.zero_grad(set_to_none=True)
        x_iter = x.detach().clone().requires_grad_()
        output = layer(x_iter)
        loss = output.square().mean()
        loss.backward()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations


if __name__ == "__main__":
    main()

