"""Compare dense MLP with a CPU/Gloo distributed MLP composition prototype."""

from __future__ import annotations

import argparse
import os
import socket

import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedColumnParallelLinear,
    DistributedRowParallelLinear,
    get_rank,
    get_world_size,
    init_distributed_from_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense MLP with a CPU/Gloo distributed MLP composition prototype."
    )
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("Distributed MLP composition demo")
        print("Run with:")
        print("  python examples/compare_distributed_mlp.py --spawn 2")
        print("This demo uses CPU/Gloo module-level linear prototypes.")
        print("GPT/model real distributed tensor parallelism is not wired yet.")
        print("No NCCL, GPU, multi-node orchestration, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()
        hidden_size = 4
        intermediate_size = 8

        if intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} must be divisible by world_size={world_size}"
            )

        if rank == 0:
            print("Distributed MLP composition demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print(f"hidden_size: {hidden_size}")
            print(f"intermediate_size: {intermediate_size}")
            print()
            print("Composition:")
            print("  DistributedColumnParallelLinear(gather_output=False)")
            print("  GELU(local shard)")
            print("  DistributedRowParallelLinear(input_is_parallel=True)")
            print()

        _barrier()
        result = _compare_forward_and_gradients(hidden_size, intermediate_size, world_size)
        _barrier()

        if rank == 0:
            print("Forward")
            print(f"  dense output shape: {result['dense_shape']}")
            print(f"  distributed output shape: {result['distributed_shape']}")
            print(f"  max abs error: {result['forward_error']:.6e}")
            print(f"  outputs close: {result['forward_close']}")
            print()
            print("Input gradient")
            print(f"  max abs error: {result['grad_error']:.6e}")
            print(f"  gradients close: {result['grad_close']}")
            print()
            print("Note:")
            print("  This is a CPU/Gloo module-level distributed MLP prototype.")
            print("  GPT/model real distributed tensor parallelism is not wired yet.")
            print("  No NCCL/GPU/multi-node/speedup claims.")
    finally:
        _destroy_process_group()


def _compare_forward_and_gradients(hidden_size: int, intermediate_size: int, world_size: int) -> dict[str, object]:
    torch.manual_seed(9301)
    dense_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
    dense_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
    activation = nn.GELU()

    dist_fc1 = DistributedColumnParallelLinear(
        in_features=hidden_size,
        out_features=intermediate_size,
        bias=True,
        gather_output=False,
    )
    dist_fc2 = DistributedRowParallelLinear(
        in_features=intermediate_size,
        out_features=hidden_size,
        bias=True,
        input_is_parallel=True,
    )
    dist_fc1.copy_from_dense_(dense_fc1)
    dist_fc2.copy_from_dense_(dense_fc2)

    dense_x = torch.randn(2, 3, hidden_size, requires_grad=True)
    dist_x = dense_x.detach().clone().requires_grad_()

    dense_y = dense_fc2(activation(dense_fc1(dense_x)))
    local_intermediate = dist_fc1(dist_x)
    local_activated = activation(local_intermediate)
    dist_y = dist_fc2(local_activated)

    dense_loss = dense_y.square().mean()
    dist_loss = dist_y.square().mean()
    dense_loss.backward()
    dist_loss.backward()

    assert dense_x.grad is not None
    assert dist_x.grad is not None

    return {
        "dense_shape": dense_y.shape,
        "distributed_shape": dist_y.shape,
        "forward_error": _max_abs_error(dense_y, dist_y),
        "forward_close": _outputs_close(dense_y, dist_y),
        "grad_error": _max_abs_error(dense_x.grad, dist_x.grad),
        "grad_close": _outputs_close(dense_x.grad, dist_x.grad),
    }


def _spawn_workers(world_size: int) -> None:
    if world_size < 1:
        raise ValueError("--spawn must be at least 1")

    import torch.multiprocessing as mp

    print(f"Spawning {world_size} local CPU/Gloo worker process(es).", flush=True)
    port = _find_free_port()
    mp.spawn(_spawn_worker, args=(world_size, port), nprocs=world_size, join=True)


def _spawn_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")
    _run_demo()


def _max_abs_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float((expected - actual).abs().max().item())


def _outputs_close(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5))


def _has_torchrun_env() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
