"""Compare dense Linear with CPU/Gloo distributed parallel linear prototypes."""

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
        description="Compare dense Linear with CPU/Gloo distributed parallel linear prototypes."
    )
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("Distributed parallel linear demo")
        print("Run with:")
        print("  python examples/compare_distributed_parallel_linear.py --spawn 2")
        print("This demo uses CPU/Gloo module-level prototypes.")
        print("GPT/model real distributed tensor parallelism is not wired yet.")
        print("No NCCL, GPU, multi-node orchestration, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()

        if rank == 0:
            print("Distributed parallel linear demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print()

        _barrier()
        _compare_column_parallel()
        _barrier()
        _compare_row_parallel()
        _barrier()

        if rank == 0:
            print()
            print("Note:")
            print("  This is CPU/Gloo distributed module-level prototyping.")
            print("  GPT/model real distributed tensor parallelism is not wired yet.")
            print("  No NCCL/GPU/multi-node/speedup claims.")
    finally:
        _destroy_process_group()


def _compare_column_parallel() -> None:
    rank = get_rank()
    world_size = get_world_size()
    batch_size = 2
    seq_len = 3
    in_features = 4
    local_out_features = 3
    out_features = local_out_features * world_size

    torch.manual_seed(8101)
    dense = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, seq_len, in_features)
    dense_y = dense(x)

    gathered_layer = DistributedColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        gather_output=True,
    )
    gathered_layer.copy_from_dense_(dense)
    gathered_y = gathered_layer(x)

    if rank == 0:
        print("ColumnParallelLinear gather_output=True")
        print(f"  dense shape: {dense_y.shape}")
        print(f"  distributed shape: {gathered_y.shape}")
        print(f"  max abs error: {_max_abs_error(dense_y, gathered_y):.6e}")
        print(f"  outputs close: {_outputs_close(dense_y, gathered_y)}")
        print()

    _barrier()

    local_layer = DistributedColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        gather_output=False,
    )
    local_layer.copy_from_dense_(dense)
    local_y = local_layer(x)
    expected_local_y = dense_y[..., local_layer.local_out_start : local_layer.local_out_end]

    _print_by_rank(
        "ColumnParallelLinear gather_output=False",
        f"  rank {rank} local output shape: {local_y.shape}",
        f"  rank {rank} local max abs error: {_max_abs_error(expected_local_y, local_y):.6e}",
    )


def _compare_row_parallel() -> None:
    rank = get_rank()
    world_size = get_world_size()
    batch_size = 2
    seq_len = 3
    local_in_features = 3
    in_features = local_in_features * world_size
    out_features = 4

    torch.manual_seed(8201)
    dense = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, seq_len, in_features)
    dense_y = dense(x)

    full_input_layer = DistributedRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        input_is_parallel=False,
    )
    full_input_layer.copy_from_dense_(dense)
    full_input_y = full_input_layer(x)

    if rank == 0:
        print("RowParallelLinear input_is_parallel=False")
        print(f"  dense shape: {dense_y.shape}")
        print(f"  distributed shape: {full_input_y.shape}")
        print(f"  max abs error: {_max_abs_error(dense_y, full_input_y):.6e}")
        print(f"  outputs close: {_outputs_close(dense_y, full_input_y)}")
        print()

    _barrier()

    parallel_input_layer = DistributedRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        input_is_parallel=True,
    )
    parallel_input_layer.copy_from_dense_(dense)
    local_x = x[..., parallel_input_layer.local_in_start : parallel_input_layer.local_in_end].contiguous()
    parallel_input_y = parallel_input_layer(local_x)

    if rank == 0:
        print("RowParallelLinear input_is_parallel=True")
        print(f"  dense shape: {dense_y.shape}")
        print(f"  distributed shape: {parallel_input_y.shape}")
        print(f"  max abs error: {_max_abs_error(dense_y, parallel_input_y):.6e}")
        print(f"  outputs close: {_outputs_close(dense_y, parallel_input_y)}")
        print()

    _barrier()
    _print_by_rank(
        "RowParallelLinear rank-local input slices",
        f"  rank {rank} local input shape: {local_x.shape}",
        f"  rank {rank} local input range: [{parallel_input_layer.local_in_start}, {parallel_input_layer.local_in_end})",
    )


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


def _print_by_rank(title: str, *lines: str) -> None:
    rank = get_rank()
    world_size = get_world_size()
    for print_rank in range(world_size):
        _barrier()
        if rank == print_rank:
            if rank == 0:
                print(title)
            for line in lines:
                print(line, flush=True)
            if rank == world_size - 1:
                print(flush=True)
    _barrier()


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
