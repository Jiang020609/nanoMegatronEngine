"""Compare dense transformer block with a CPU/Gloo distributed block prototype."""

from __future__ import annotations

import argparse
import os
import socket

import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_transformer_block import DistributedTransformerBlock
from nano_megatron_engine.model.transformer import TransformerBlock
from nano_megatron_engine.parallel import get_rank, get_world_size, init_distributed_from_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense transformer block with a CPU/Gloo distributed block prototype."
    )
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("Distributed transformer block demo")
        print("Run with:")
        print("  python examples/compare_distributed_transformer_block.py --spawn 2")
        print("This demo uses CPU/Gloo module-level transformer block prototypes.")
        print("GPT/model real distributed tensor parallelism is not wired yet.")
        print("No NCCL, GPU, multi-node orchestration, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()
        hidden_size = 8
        num_heads = 4
        mlp_hidden_size = 32
        block_size = 8

        if num_heads % world_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by world_size={world_size}")
        if mlp_hidden_size % world_size != 0:
            raise ValueError(f"mlp_hidden_size={mlp_hidden_size} must be divisible by world_size={world_size}")

        result = _compare_block(hidden_size, num_heads, mlp_hidden_size, block_size)

        if rank == 0:
            print("Distributed transformer block demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print(f"hidden_size: {hidden_size}")
            print(f"num_heads: {num_heads}")
            print(f"local_heads: {result['local_heads']}")
            print(f"mlp_hidden_size: {mlp_hidden_size}")
            print(f"block_size: {block_size}")
            print()
            print("Forward")
            print(f"  dense output shape: {result['dense_shape']}")
            print(f"  distributed output shape: {result['distributed_shape']}")
            print(f"  max abs error: {result['forward_error']:.6e}")
            print(f"  outputs close: {result['forward_close']}")
            print()
            print("Note:")
            print("  This is a CPU/Gloo module-level distributed transformer block prototype.")
            print("  GPT/model real distributed tensor parallelism is not wired yet.")
            print("  No NCCL/GPU/multi-node/speedup claims.")
    finally:
        _destroy_process_group()


def _compare_block(hidden_size: int, num_heads: int, mlp_hidden_size: int, block_size: int) -> dict[str, object]:
    torch.manual_seed(12101)
    config = GPTConfig(
        vocab_size=32,
        block_size=block_size,
        n_layer=1,
        n_head=num_heads,
        n_embd=hidden_size,
        dropout=0.0,
    )
    dense = TransformerBlock(config)
    dense.eval()
    distributed = DistributedTransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=block_size,
        mlp_hidden_size=mlp_hidden_size,
        bias=True,
        dropout=0.0,
    )
    distributed.eval()
    distributed.copy_from_dense_(dense)

    x = torch.randn(2, 5, hidden_size)
    dense_y = dense(x)
    distributed_y = distributed(x)

    return {
        "local_heads": distributed.local_heads,
        "dense_shape": dense_y.shape,
        "distributed_shape": distributed_y.shape,
        "forward_error": _max_abs_error(dense_y, distributed_y),
        "forward_close": _outputs_close(dense_y, distributed_y),
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


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
