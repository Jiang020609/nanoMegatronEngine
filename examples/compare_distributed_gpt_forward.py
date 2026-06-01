"""Compare dense GPT forward with a CPU/Gloo distributed GPT prototype."""

from __future__ import annotations

import argparse
import os
import socket

import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import get_rank, get_world_size, init_distributed_from_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense GPT forward with a CPU/Gloo distributed GPT prototype."
    )
    parser.add_argument("--spawn", type=int, default=0, help="spawn this many local CPU/Gloo worker processes")
    args = parser.parse_args()

    if args.spawn < 0:
        parser.error("--spawn must be non-negative")
    if args.spawn > 0:
        _spawn_workers(args.spawn)
        return

    if not _has_torchrun_env():
        print("Distributed GPT forward demo")
        print("Run with:")
        print("  python examples/compare_distributed_gpt_forward.py --spawn 2")
        print("This demo uses CPU/Gloo module-level distributed GPT forward prototypes.")
        print("Training loops and real distributed GPT TP are not wired yet.")
        print("The optimizer section is a one-step smoke check only.")
        print("No NCCL, GPU, multi-node orchestration, or speedup claims.")
        return

    _run_demo()


def _run_demo() -> None:
    init_distributed_from_env("gloo")
    try:
        rank = get_rank()
        world_size = get_world_size()
        if world_size != 2:
            raise ValueError("this tiny demo expects --spawn 2")

        result = _compare_gpt_forward(world_size)

        if rank == 0:
            print("Distributed GPT forward demo")
            print("backend: gloo")
            print(f"world_size: {world_size}")
            print(f"vocab_size: {result['vocab_size']}")
            print(f"block_size: {result['block_size']}")
            print(f"n_layer: {result['n_layer']}")
            print(f"n_head: {result['n_head']}")
            print(f"n_embd: {result['n_embd']}")
            print()
            print("Local shard summaries")
            for summary in result["shard_summaries"]:
                print(_format_shard_summary(summary))
            print()
            print("Forward")
            print(f"  dense logits shape: {result['dense_shape']}")
            print(f"  distributed logits shape: {result['distributed_shape']}")
            print(f"  max abs error: {result['logits_error']:.6e}")
            print(f"  logits close: {result['logits_close']}")
            print(f"  loss max abs error: {result['loss_error']:.6e}")
            print(f"  loss close: {result['loss_close']}")
            print()
            print("Backward smoke")
            print("  loss backward completed on every rank")
            print(f"  trainable gradients finite: {result['grads_finite']}")
            print(f"  rank 0 tied vocab grad shape: {result['tied_vocab_grad_shape']}")
            print(f"  rank 0 tied vocab grad nonzero: {result['tied_vocab_grad_nonzero']}")
            print()
            print("Optimizer step smoke")
            print(f"  one SGD step completed: {result['optimizer_step_completed']}")
            print(f"  replicated gradients synchronized: {result['replicated_gradients_synchronized']}")
            print(f"  local parameters changed: {result['parameters_changed']}")
            print(f"  local parameters finite after step: {result['parameters_finite_after_step']}")
            print(f"  post-step loss finite: {result['post_step_loss_finite']}")
            print()
            print("Note:")
            print("  This is a CPU/Gloo distributed GPT forward prototype.")
            print("  The backward and optimizer sections check local plumbing only.")
            print("  Replicated gradient synchronization is explicit and prototype-local.")
            print("  Full dense-equivalent distributed GPT training is not claimed yet.")
            print("  Real distributed GPT TP is not wired into the main GPTModel path.")
            print("  No NCCL/GPU/multi-node/speedup claims.")
    finally:
        _destroy_process_group()


def _compare_gpt_forward(world_size: int) -> dict[str, object]:
    torch.manual_seed(13101)
    dense_config = _dense_config()
    distributed_config = _distributed_config(world_size)
    dense = GPTModel(dense_config)
    dense.eval()
    distributed = DistributedGPTModel(distributed_config)
    distributed.eval()
    distributed.copy_from_dense_(dense)
    shard_summaries = _gather_shard_summaries(distributed.local_shard_summary())

    input_ids = torch.tensor(
        [
            [0, 1, 16, 31, 4],
            [17, 2, 30, 8, 15],
        ]
    )
    targets = torch.tensor(
        [
            [1, 16, 31, 4, 5],
            [2, 30, 8, 15, 0],
        ]
    )
    dense_logits, dense_loss = dense(input_ids, targets)
    distributed_logits, distributed_loss = distributed(input_ids, targets)
    assert dense_loss is not None
    assert distributed_loss is not None
    dense_loss.backward()
    distributed_loss.backward()

    _assert_all_trainable_grads_are_finite(distributed)
    start = distributed.token_embedding.local_vocab_start
    end = distributed.token_embedding.local_vocab_end
    assert dense.token_embedding.weight.grad is not None
    assert distributed.token_embedding.weight_shards[0].grad is not None
    dense_tied_vocab_grad = dense.token_embedding.weight.grad[start:end]
    distributed_tied_vocab_grad = distributed.token_embedding.weight_shards[0].grad
    if distributed_tied_vocab_grad.shape != dense_tied_vocab_grad.shape:
        raise AssertionError("distributed GPT tied vocab gradient shard shape does not match the dense slice shape")
    optimizer_result = _run_optimizer_step_smoke(distributed, input_ids, targets)

    return {
        "vocab_size": dense_config.vocab_size,
        "block_size": dense_config.block_size,
        "n_layer": dense_config.n_layer,
        "n_head": dense_config.n_head,
        "n_embd": dense_config.n_embd,
        "shard_summaries": shard_summaries,
        "dense_shape": dense_logits.shape,
        "distributed_shape": distributed_logits.shape,
        "logits_error": _max_abs_error(dense_logits, distributed_logits),
        "logits_close": _outputs_close(dense_logits, distributed_logits),
        "loss_error": _max_abs_error(dense_loss, distributed_loss),
        "loss_close": _outputs_close(dense_loss, distributed_loss),
        "grads_finite": _all_trainable_grads_are_finite(distributed),
        "tied_vocab_grad_shape": distributed_tied_vocab_grad.shape,
        "tied_vocab_grad_nonzero": bool(distributed_tied_vocab_grad.abs().sum().item() > 0.0),
        **optimizer_result,
    }


def _dense_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=1,
    )


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


def _assert_all_trainable_grads_are_finite(model: torch.nn.Module) -> None:
    if not _all_trainable_grads_are_finite(model):
        raise AssertionError("distributed GPT prototype produced missing or non-finite gradients")


def _all_trainable_grads_are_finite(model: torch.nn.Module) -> bool:
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not grads:
        return False
    return all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def _gather_shard_summaries(summary: dict[str, object]) -> list[dict[str, object]]:
    import torch.distributed as dist

    summaries: list[dict[str, object] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(summaries, summary)
    return [item for item in summaries if item is not None]


def _format_shard_summary(summary: dict[str, object]) -> str:
    token_embedding = summary["token_embedding"]
    lm_head = summary["lm_head"]
    blocks = summary["blocks"]
    first_block = blocks[0]
    attention = first_block["attention"]
    mlp = first_block["mlp"]
    return (
        f"  rank {summary['rank']}: params={summary['local_parameter_count']} "
        f"token_vocab={token_embedding['vocab_range']} token_weight={token_embedding['weight_shape']} "
        f"lm_head_weight={lm_head['weight_shape']} tied={lm_head['tied_to_token_embedding']} "
        f"blocks={len(blocks)} block0_qkv={attention['qkv_weight_shape']} "
        f"block0_fc1={mlp['fc1_weight_shape']} block0_fc2={mlp['fc2_weight_shape']}"
    )


def _run_optimizer_step_smoke(
    model: DistributedGPTModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    synchronized = model.synchronize_replicated_gradients_()
    if not synchronized:
        raise AssertionError("distributed GPT optimizer smoke did not synchronize replicated gradients")
    optimizer.step()
    after = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
    parameters_changed = bool(any(not torch.equal(old, new) for old, new in zip(before, after)))
    parameters_finite = bool(all(torch.isfinite(parameter).all() for parameter in after))
    with torch.no_grad():
        _, post_step_loss = model(input_ids, targets)
    post_step_loss_finite = bool(post_step_loss is not None and torch.isfinite(post_step_loss))
    if not parameters_changed:
        raise AssertionError("distributed GPT optimizer smoke step did not change any local trainable parameter")
    if not parameters_finite:
        raise AssertionError("distributed GPT optimizer smoke step produced non-finite local parameters")
    if not post_step_loss_finite:
        raise AssertionError("distributed GPT optimizer smoke step produced a non-finite post-step loss")
    return {
        "optimizer_step_completed": True,
        "replicated_gradients_synchronized": len(synchronized),
        "parameters_changed": parameters_changed,
        "parameters_finite_after_step": parameters_finite,
        "post_step_loss_finite": post_step_loss_finite,
    }


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
