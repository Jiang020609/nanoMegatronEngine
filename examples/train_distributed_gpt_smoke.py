"""Run a tiny CPU/Gloo distributed GPT optimizer-loop smoke test."""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.parallel import get_rank, get_world_size, init_distributed_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny CPU/Gloo distributed GPT smoke training loop.")
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    parser.add_argument("--steps", type=int, default=2, help="number of tiny optimizer smoke steps")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.spawn > 0:
        _spawn_workers(args.spawn, args.steps)
        return

    if not _has_torchrun_env():
        print("Distributed GPT smoke training demo")
        print("Run with:")
        print("  python examples/train_distributed_gpt_smoke.py --spawn 2")
        print("This is a CPU/Gloo prototype-local optimizer-loop smoke test.")
        print("It is not wired into GPTModel or the single-device Trainer.")
        print("CUDA/NCCL smoke validation lives in compare_distributed_gpt_nccl.py.")
        print("No multi-node orchestration, speedup, or convergence claims.")
        return

    _run_demo(args.steps)


def _run_demo(steps: int) -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()
        if world_size != 2:
            raise ValueError("this tiny demo expects --spawn 2")

        result = _run_smoke_training_loop(world_size, steps)

        if rank == 0:
            print("Distributed GPT smoke training demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print(f"steps: {steps}")
            print(f"vocab_size: {result['vocab_size']}")
            print(f"block_size: {result['block_size']}")
            print(f"n_layer: {result['n_layer']}")
            print(f"n_head: {result['n_head']}")
            print(f"n_embd: {result['n_embd']}")
            print(f"local parameters per rank: {result['local_parameter_count']}")
            print()
            for step, loss in result["losses"]:
                print(f"step={step:02d} loss={loss:.6f}")
            print()
            print("Smoke checks")
            print(f"  replicated gradients synchronized each step: {result['replicated_gradients_synchronized']}")
            print(f"  local parameters finite after training: {result['parameters_finite']}")
            print(f"  elapsed seconds: {result['elapsed_seconds']:.4f}")
            print()
            print("Note:")
            print("  This is a CPU/Gloo prototype-local optimizer-loop smoke test.")
            print("  It does not claim dense-equivalent distributed training semantics.")
            print("  CUDA/NCCL smoke validation lives in compare_distributed_gpt_nccl.py.")
            print("  GPTModel, Trainer, multi-node orchestration, and speedup paths are not wired.")
    finally:
        _destroy_process_group()


def _run_smoke_training_loop(world_size: int, steps: int) -> dict[str, object]:
    torch.manual_seed(13201)
    config = _distributed_config(world_size)
    model = DistributedGPTModel(config)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    input_ids, targets = _make_synthetic_batch(config)

    losses = []
    synchronized_counts = []
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        if loss is None or not torch.isfinite(loss):
            raise AssertionError("distributed GPT smoke training loop produced a non-finite loss")
        loss.backward()
        _assert_all_trainable_grads_are_finite(model)
        synchronized = model.synchronize_replicated_gradients_()
        if not synchronized:
            raise AssertionError("distributed GPT smoke training loop did not synchronize replicated gradients")
        optimizer.step()
        losses.append((step, float(loss.detach().item())))
        synchronized_counts.append(len(synchronized))

    elapsed = time.perf_counter() - start_time
    parameters_finite = bool(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
    if not parameters_finite:
        raise AssertionError("distributed GPT smoke training loop produced non-finite parameters")

    return {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "local_parameter_count": model.num_parameters(),
        "losses": losses,
        "replicated_gradients_synchronized": synchronized_counts,
        "parameters_finite": parameters_finite,
        "elapsed_seconds": elapsed,
    }


def _distributed_config(world_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=world_size,
    )


def _make_synthetic_batch(config: GPTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [0, 1, 16, 31, 4],
            [17, 2, 30, 8, 15],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [1, 16, 31, 4, 5],
            [2, 30, 8, 15, 0],
        ],
        dtype=torch.long,
    )
    if input_ids.max().item() >= config.vocab_size:
        raise AssertionError("synthetic input token id exceeds vocab_size")
    return input_ids, targets


def _spawn_workers(world_size: int, steps: int) -> None:
    if world_size < 1:
        raise ValueError("--spawn must be at least 1")

    import torch.multiprocessing as mp

    print(f"Spawning {world_size} local CPU/Gloo worker process(es).", flush=True)
    port = _find_free_port()
    mp.spawn(_spawn_worker, args=(world_size, port, steps), nprocs=world_size, join=True)


def _spawn_worker(rank: int, world_size: int, port: int, steps: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")
    _run_demo(steps)


def _assert_all_trainable_grads_are_finite(model: torch.nn.Module) -> None:
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not grads:
        raise AssertionError("distributed GPT smoke training loop did not expose trainable parameters")
    if not all(grad is not None and torch.isfinite(grad).all() for grad in grads):
        raise AssertionError("distributed GPT smoke training loop produced missing or non-finite gradients")


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
