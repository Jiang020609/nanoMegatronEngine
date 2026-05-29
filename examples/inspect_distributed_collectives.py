"""Inspect optional CPU/Gloo distributed collective wrappers."""

from __future__ import annotations

import argparse
import os
import socket

import torch

from nano_megatron_engine.parallel import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_rank,
    get_world_size,
    init_distributed_from_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect optional CPU/Gloo distributed collective wrappers.")
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("CPU/Gloo distributed collective inspection")
        print("Run with:")
        print("  python examples/inspect_distributed_collectives.py --spawn 2")
        print("  torchrun --standalone --nproc_per_node=2 examples/inspect_distributed_collectives.py")
        print("This example uses torch.distributed with Gloo on CPU; no NCCL, GPU, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()

        local = torch.full((2, 3), float(rank + 1))
        reduced = distributed_all_reduce_sum(local)

        gather_input = torch.full((2, rank + 2), float(rank + 1))
        gathered = distributed_all_gather(gather_input, dim=-1)

        base = torch.arange(10, dtype=torch.float32).view(2, 5)
        scattered = distributed_reduce_scatter_sum(base + rank, dim=-1)

        print(f"rank={rank} world_size={world_size}")
        print("backend=gloo device=cpu")
        print("This is an optional wrapper demo, not real GPT tensor parallelism and not a speedup claim.")
        print(f"all_reduce_sum={reduced.tolist()}")
        print(f"all_gather_shape={tuple(gathered.shape)} all_gather={gathered.tolist()}")
        print(f"reduce_scatter_shape={tuple(scattered.shape)} reduce_scatter={scattered.tolist()}")
    finally:
        _destroy_process_group()


def _spawn_workers(world_size: int) -> None:
    if world_size < 1:
        raise ValueError("--spawn must be at least 1")

    import torch.multiprocessing as mp

    port = _find_free_port()
    mp.spawn(_spawn_worker, args=(world_size, port), nprocs=world_size, join=True)


def _spawn_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")
    _run_demo()


def _has_torchrun_env() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
