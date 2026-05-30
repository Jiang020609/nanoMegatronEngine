"""Compare dense vocab modules with CPU/Gloo distributed vocab-parallel prototypes."""

from __future__ import annotations

import argparse
import os
import socket

import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedRankLocalCollectives,
    VocabParallelEmbedding,
    VocabParallelLMHead,
    get_rank,
    get_world_size,
    init_distributed_from_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense vocab modules with CPU/Gloo distributed vocab-parallel prototypes."
    )
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("Distributed vocab-parallel demo")
        print("Run with:")
        print("  python examples/compare_distributed_vocab_parallel.py --spawn 2")
        print("This demo uses CPU/Gloo module-level vocab prototypes.")
        print("GPT/model real distributed tensor parallelism is not wired yet.")
        print("Distributed vocab partitions are strict divisible for now.")
        print("No NCCL, GPU, multi-node orchestration, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()
        vocab_size = 8
        hidden_size = 4

        if vocab_size % world_size != 0:
            raise ValueError(f"vocab_size={vocab_size} must be divisible by world_size={world_size}")

        if rank == 0:
            print("Distributed vocab-parallel demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print(f"vocab_size: {vocab_size}")
            print(f"hidden_size: {hidden_size}")
            print()

        _barrier()
        _print_vocab_ranges(vocab_size)
        _barrier()
        _compare_embedding(vocab_size, hidden_size, world_size)
        _barrier()
        _compare_lm_head(vocab_size, hidden_size, world_size)
        _barrier()

        if rank == 0:
            print("Note:")
            print("  This is CPU/Gloo distributed module-level prototyping.")
            print("  GPT/model real distributed tensor parallelism is not wired yet.")
            print("  Distributed vocab partitions are strict divisible for now.")
            print("  No NCCL/GPU/multi-node/speedup claims.")
    finally:
        _destroy_process_group()


def _print_vocab_ranges(vocab_size: int) -> None:
    world_size = get_world_size()
    local_size = vocab_size // world_size
    rank = get_rank()
    start = rank * local_size
    end = start + local_size
    _print_by_rank(
        "Vocab shard ranges",
        f"  rank {rank} vocab range: [{start}, {end})",
    )


def _compare_embedding(vocab_size: int, hidden_size: int, world_size: int) -> None:
    rank = get_rank()
    torch.manual_seed(9201)
    dense_embedding = nn.Embedding(vocab_size, hidden_size)
    tokens = torch.tensor(
        [
            [0, 1, 4],
            [5, 2, 7],
        ]
    )
    distributed_embedding = VocabParallelEmbedding.from_embedding(
        dense_embedding,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )

    dense_output = dense_embedding(tokens)
    distributed_output = distributed_embedding(tokens)

    if rank == 0:
        print("VocabParallelEmbedding")
        print(f"  dense output shape: {dense_output.shape}")
        print(f"  distributed output shape: {distributed_output.shape}")
        print(f"  max abs error: {_max_abs_error(dense_output, distributed_output):.6e}")
        print(f"  outputs close: {_outputs_close(dense_output, distributed_output)}")
        print()


def _compare_lm_head(vocab_size: int, hidden_size: int, world_size: int) -> None:
    rank = get_rank()
    torch.manual_seed(9202)
    dense_lm_head = nn.Linear(hidden_size, vocab_size, bias=True)
    hidden = torch.randn(2, 3, hidden_size)
    distributed_lm_head = VocabParallelLMHead.from_linear(
        dense_lm_head,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )

    dense_logits = dense_lm_head(hidden)
    distributed_logits = distributed_lm_head(hidden)

    if rank == 0:
        print("VocabParallelLMHead")
        print(f"  dense logits shape: {dense_logits.shape}")
        print(f"  distributed logits shape: {distributed_logits.shape}")
        print(f"  max abs error: {_max_abs_error(dense_logits, distributed_logits):.6e}")
        print(f"  outputs close: {_outputs_close(dense_logits, distributed_logits)}")
        print()

    _barrier()
    _compare_lm_head_input_gradients(vocab_size, hidden_size, world_size)


def _compare_lm_head_input_gradients(vocab_size: int, hidden_size: int, world_size: int) -> None:
    rank = get_rank()
    torch.manual_seed(9203)
    dense_lm_head = nn.Linear(hidden_size, vocab_size, bias=True)
    distributed_lm_head = VocabParallelLMHead.from_linear(
        dense_lm_head,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )

    dense_hidden = torch.randn(2, 3, hidden_size, requires_grad=True)
    distributed_hidden = dense_hidden.detach().clone().requires_grad_()

    dense_loss = dense_lm_head(dense_hidden).square().mean()
    distributed_loss = distributed_lm_head(distributed_hidden).square().mean()
    dense_loss.backward()
    distributed_loss.backward()

    assert dense_hidden.grad is not None
    assert distributed_hidden.grad is not None
    if rank == 0:
        print("LM head input gradient")
        print(f"  max abs error: {_max_abs_error(dense_hidden.grad, distributed_hidden.grad):.6e}")
        print(f"  gradients close: {_outputs_close(dense_hidden.grad, distributed_hidden.grad)}")
        print()


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
