"""Run a tiny CUDA/NCCL distributed GPT prototype smoke comparison."""

from __future__ import annotations

import argparse
import os

import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import get_backend, get_rank, get_world_size, init_distributed_from_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny CUDA/NCCL distributed GPT prototype smoke comparison."
    )
    parser.add_argument(
        "--preset",
        choices=("tiny", "small"),
        default="tiny",
        help="model shape preset; small is still a smoke test but exercises wider tensors",
    )
    parser.add_argument("--seed", type=int, default=13101, help="deterministic seed for model initialization")
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="disable attention and MLP projection biases in dense and distributed GPT",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="print checks without failing on close/finiteness mismatches",
    )
    parser.set_defaults(strict=True)
    args = parser.parse_args()

    if not _has_torchrun_env():
        print("CUDA/NCCL distributed GPT smoke demo")
        print("Run with torchrun on a CUDA PyTorch build with NCCL, for example:")
        print("  torchrun --standalone --nproc_per_node=2 examples/compare_distributed_gpt_nccl.py")
        print("  torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_nccl.py")
        print("  torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_nccl.py --preset small")
        print("  torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_nccl.py --preset small --no-bias")
        print("This is an isolated distributed GPT prototype smoke path.")
        print("Strict validation is enabled by default; use --no-strict to print without failing.")
        print("The main GPTModel path is not wired to real distributed TP.")
        print("No multi-node orchestration or speedup claims.")
        return

    _run_demo(args)


def _run_demo(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/NCCL distributed GPT smoke requires torch.cuda.is_available()")
    if not dist.is_available() or not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available():
        raise RuntimeError("CUDA/NCCL distributed GPT smoke requires a PyTorch build with NCCL support")

    local_rank = _local_rank()
    if local_rank >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA device count {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    completed = False
    init_distributed_from_env("nccl")
    try:
        rank = get_rank()
        world_size = get_world_size()
        backend = get_backend()
        use_bias = not args.no_bias
        dense_config = _dense_config(args.preset, bias=use_bias)
        if dense_config.n_head % world_size != 0 or dense_config.vocab_size % world_size != 0:
            raise ValueError(
                "this CUDA/NCCL demo expects world_size to divide "
                f"n_head={dense_config.n_head} and vocab_size={dense_config.vocab_size}; "
                "try --nproc_per_node=2 or --nproc_per_node=4 with the default presets"
            )

        result = _compare_gpt_forward(world_size, device, preset=args.preset, seed=args.seed, bias=use_bias)
        torch.cuda.synchronize(device)

        if rank == 0:
            print("CUDA/NCCL distributed GPT smoke demo")
            print(f"backend: {backend}")
            print(f"world_size: {world_size}")
            print(f"cuda devices visible: {torch.cuda.device_count()}")
            print(f"rank 0 device: {device}")
            print(f"preset: {result['preset']}")
            print(f"seed: {result['seed']}")
            print(f"strict validation: {args.strict}")
            print(f"vocab_size: {result['vocab_size']}")
            print(f"block_size: {result['block_size']}")
            print(f"n_layer: {result['n_layer']}")
            print(f"n_head: {result['n_head']}")
            print(f"n_embd: {result['n_embd']}")
            print(f"bias: {result['bias']}")
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
            print(f"  activation checkpoint backward smoke: {result['activation_checkpoint_backward_smoke']}")
            print()
            print("Strict checks")
            print(f"  all strict checks passed: {_strict_checks_pass(result)}")
            print()
            print("Note:")
            print("  This is a CUDA/NCCL smoke path for the isolated distributed GPT prototype.")
            print("  The backward and optimizer sections check local plumbing only.")
            print("  Replicated gradient synchronization is explicit and prototype-local.")
            print("  Full dense-equivalent distributed GPT training is not claimed yet.")
            print("  Real distributed GPT TP is not wired into the main GPTModel path.")
            print("  No multi-node orchestration or speedup claims.")

        if args.strict:
            _assert_strict_result(result)
        _synchronize_and_barrier(device)
        completed = True
    finally:
        if dist.is_available() and dist.is_initialized():
            if completed:
                torch.cuda.synchronize(device)
            dist.destroy_process_group()


def _compare_gpt_forward(
    world_size: int,
    device: torch.device,
    preset: str,
    seed: int,
    bias: bool,
) -> dict[str, object]:
    _make_cuda_math_deterministic_enough()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dense_config = _dense_config(preset, bias=bias)
    distributed_config = _distributed_config(world_size, preset, bias=bias)
    dense = GPTModel(dense_config).to(device)
    dense.eval()
    distributed = DistributedGPTModel(distributed_config).to(device)
    distributed.eval()
    distributed.copy_from_dense_(dense)
    shard_summaries = _gather_shard_summaries(distributed.local_shard_summary())

    input_ids, targets = _make_synthetic_batch(dense_config, device)

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
    activation_checkpoint_result = _run_activation_checkpointing_smoke(
        world_size,
        input_ids,
        targets,
        device,
        preset=preset,
        seed=seed + 1,
        bias=bias,
    )

    logits_close = _outputs_close(dense_logits, distributed_logits)
    loss_close = _outputs_close(dense_loss, distributed_loss)

    return {
        "preset": preset,
        "seed": seed,
        "vocab_size": dense_config.vocab_size,
        "block_size": dense_config.block_size,
        "n_layer": dense_config.n_layer,
        "n_head": dense_config.n_head,
        "n_embd": dense_config.n_embd,
        "bias": dense_config.bias,
        "shard_summaries": shard_summaries,
        "dense_shape": dense_logits.shape,
        "distributed_shape": distributed_logits.shape,
        "logits_error": _max_abs_error(dense_logits, distributed_logits),
        "logits_close": logits_close,
        "logits_finite": _tensor_is_finite(dense_logits) and _tensor_is_finite(distributed_logits),
        "loss_error": _max_abs_error(dense_loss, distributed_loss),
        "loss_close": loss_close,
        "loss_finite": _tensor_is_finite(dense_loss) and _tensor_is_finite(distributed_loss),
        "grads_finite": _all_trainable_grads_are_finite(distributed),
        "tied_vocab_grad_shape": distributed_tied_vocab_grad.shape,
        "tied_vocab_grad_nonzero": bool(distributed_tied_vocab_grad.abs().sum().item() > 0.0),
        **optimizer_result,
        **activation_checkpoint_result,
    }


def _dense_config(preset: str, bias: bool = True) -> GPTConfig:
    if preset == "small":
        return GPTConfig(
            vocab_size=128,
            block_size=16,
            n_layer=2,
            n_head=8,
            n_embd=64,
            bias=bias,
            dropout=0.0,
            tensor_parallel_size=1,
        )
    if preset != "tiny":
        raise ValueError(f"unknown CUDA/NCCL GPT smoke preset {preset!r}")
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        bias=bias,
        dropout=0.0,
        tensor_parallel_size=1,
    )


def _distributed_config(world_size: int, preset: str, bias: bool = True) -> GPTConfig:
    config = _dense_config(preset, bias=bias)
    return GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=0.0,
        tensor_parallel_size=world_size,
    )


def _distributed_checkpoint_config(world_size: int, preset: str, bias: bool = True) -> GPTConfig:
    config = _dense_config(preset, bias=bias)
    return GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=0.0,
        use_activation_checkpointing=True,
        tensor_parallel_size=world_size,
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


def _run_activation_checkpointing_smoke(
    world_size: int,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    preset: str,
    seed: int,
    bias: bool,
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DistributedGPTModel(_distributed_checkpoint_config(world_size, preset, bias=bias)).to(device)
    model.train()
    _, loss = model(input_ids, targets)
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("distributed GPT activation checkpoint smoke produced a non-finite loss")
    loss.backward()
    _assert_all_trainable_grads_are_finite(model)
    synchronized = model.synchronize_replicated_gradients_()
    if not synchronized:
        raise AssertionError("distributed GPT activation checkpoint smoke did not synchronize replicated gradients")
    return {"activation_checkpoint_backward_smoke": True}


def _make_synthetic_batch(config: GPTConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(5, config.block_size)
    input_ids = torch.arange(2 * seq_len, dtype=torch.long, device=device).view(2, seq_len)
    input_ids = (input_ids * 7 + 3) % config.vocab_size
    targets = (input_ids + 1) % config.vocab_size
    return input_ids, targets


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


def _max_abs_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float((expected - actual).abs().max().item())


def _outputs_close(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return bool(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _synchronize_and_barrier(device: torch.device) -> None:
    import torch.distributed as dist

    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)


def _strict_checks_pass(result: dict[str, object]) -> bool:
    return all(
        bool(result[name])
        for name in (
            "logits_close",
            "logits_finite",
            "loss_close",
            "loss_finite",
            "grads_finite",
            "tied_vocab_grad_nonzero",
            "optimizer_step_completed",
            "parameters_changed",
            "parameters_finite_after_step",
            "post_step_loss_finite",
            "activation_checkpoint_backward_smoke",
        )
    )


def _assert_strict_result(result: dict[str, object]) -> None:
    failed = [
        name
        for name in (
            "logits_close",
            "logits_finite",
            "loss_close",
            "loss_finite",
            "grads_finite",
            "tied_vocab_grad_nonzero",
            "optimizer_step_completed",
            "parameters_changed",
            "parameters_finite_after_step",
            "post_step_loss_finite",
            "activation_checkpoint_backward_smoke",
        )
        if not bool(result[name])
    ]
    if failed:
        raise AssertionError(
            "strict CUDA/NCCL distributed GPT smoke failed checks: "
            + ", ".join(failed)
            + f"; logits_error={result['logits_error']}, loss_error={result['loss_error']}"
        )


def _assert_all_trainable_grads_are_finite(model: torch.nn.Module) -> None:
    if not _all_trainable_grads_are_finite(model):
        raise AssertionError("distributed GPT prototype produced missing or non-finite gradients")


def _all_trainable_grads_are_finite(model: torch.nn.Module) -> bool:
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not grads:
        return False
    return all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def _make_cuda_math_deterministic_enough() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _local_rank() -> int:
    raw_local_rank = os.environ.get("LOCAL_RANK", os.environ["RANK"])
    try:
        return int(raw_local_rank)
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw_local_rank!r}") from exc


def _has_torchrun_env() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


if __name__ == "__main__":
    main()
