"""Run a CUDA/NCCL distributed GPT multi-step training equivalence comparison."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from compare_distributed_gpt_gradients_nccl import (
    _ATOL,
    _RTOL,
    _add_replicated_gradient_checks,
    _add_replicated_parameter_checks,
    _add_sharded_gradient_checks,
    _add_sharded_parameter_checks,
    _dense_config,
    _distributed_config,
    _format_check,
    _gather_checks,
    _has_torchrun_env,
    _local_rank,
    _local_qkv_slice,
    _make_boolean_check,
    _make_cuda_math_deterministic_enough,
    _make_tensor_check,
    _synchronize_and_barrier,
)
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import get_backend, get_rank, get_world_size, init_distributed_from_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a short dense GPT training loop and optimizer state against "
            "CUDA/NCCL distributed GPT local parameter shards."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("tiny", "small"),
        default="tiny",
        help="model shape preset; small is the intended 4-GPU A800 validation shape",
    )
    parser.add_argument("--seed", type=int, default=13301, help="deterministic seed for model initialization")
    parser.add_argument("--steps", type=int, default=5, help="number of deterministic training steps to compare")
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="optimizer to compare; AdamW also checks exp_avg and exp_avg_sq state shards",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the update checks")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for the selected optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        metavar=("BETA1", "BETA2"),
        help="AdamW beta coefficients",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--max-report", type=int, default=24, help="maximum failed/largest-error checks to print")
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="print training checks without failing on mismatches",
    )
    parser.set_defaults(strict=True)
    args = parser.parse_args()

    if not _has_torchrun_env():
        print("CUDA/NCCL distributed GPT multi-step training equivalence demo")
        print("Run with torchrun on a CUDA PyTorch build with NCCL, for example:")
        print(
            "  torchrun --standalone --nproc_per_node=4 "
            "examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5"
        )
        print(
            "  torchrun --standalone --nproc_per_node=4 "
            "examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5 "
            "--optimizer adamw --weight-decay 0.01"
        )
        print("Strict validation is enabled by default; use --no-strict to print without failing.")
        print("This compares a short optimizer loop against dense GPT slices after every step.")
        print("The main GPTModel path is not wired to real distributed TP.")
        print("No multi-node orchestration or speedup claims.")
        return

    _run_demo(args)


def _run_demo(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/NCCL distributed GPT training equivalence requires torch.cuda.is_available()")
    if not dist.is_available() or not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available():
        raise RuntimeError("CUDA/NCCL distributed GPT training equivalence requires a PyTorch build with NCCL support")

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
        dense_config = _dense_config(args.preset)
        if dense_config.n_head % world_size != 0 or dense_config.vocab_size % world_size != 0:
            raise ValueError(
                "this CUDA/NCCL training equivalence demo expects world_size to divide "
                f"n_head={dense_config.n_head} and vocab_size={dense_config.vocab_size}; "
                "try --nproc_per_node=2 or --nproc_per_node=4 with the default presets"
            )

        result = _compare_training_equivalence(
            world_size,
            device,
            preset=args.preset,
            seed=args.seed,
            steps=args.steps,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            eps=args.eps,
        )
        torch.cuda.synchronize(device)

        if rank == 0:
            _print_report(result, backend=backend, device=device, args=args)

        _synchronize_and_barrier(device)
        if args.strict:
            _assert_strict_training_result(result)
        _synchronize_and_barrier(device)
        completed = True
    finally:
        if dist.is_available() and dist.is_initialized():
            if completed:
                torch.cuda.synchronize(device)
            dist.destroy_process_group()


def _compare_training_equivalence(
    world_size: int,
    device: torch.device,
    preset: str,
    seed: int,
    steps: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> dict[str, object]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    _validate_optimizer_args(optimizer_name, learning_rate, weight_decay, betas, eps)

    _make_cuda_math_deterministic_enough()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dense_config = _dense_config(preset)
    distributed_config = _distributed_config(world_size, preset)
    dense = GPTModel(dense_config).to(device)
    dense.train()
    distributed = DistributedGPTModel(distributed_config).to(device)
    distributed.train()
    distributed.copy_from_dense_(dense)

    dense_optimizer, distributed_optimizer = _make_optimizers(
        dense,
        distributed,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    rank = get_rank()
    expected_synchronized = set(distributed.replicated_parameter_names())
    all_checks: list[dict[str, object]] = []
    step_summaries: list[dict[str, object]] = []

    for step in range(steps):
        input_ids, targets = _make_synthetic_training_batch(dense_config, device, step)
        dense.zero_grad(set_to_none=True)
        distributed.zero_grad(set_to_none=True)

        dense_logits, dense_loss = dense(input_ids, targets)
        distributed_logits, distributed_loss = distributed(input_ids, targets)
        assert dense_loss is not None
        assert distributed_loss is not None

        dense_loss.backward()
        distributed_loss.backward()

        local_checks: list[dict[str, object]] = [
            _make_tensor_check(rank, "forward.logits", "forward", dense_logits, distributed_logits),
            _make_tensor_check(rank, "forward.loss", "forward", dense_loss, distributed_loss),
        ]
        _add_sharded_gradient_checks(local_checks, rank, dense, distributed)

        synchronized = distributed.synchronize_replicated_gradients_()
        local_checks.append(
            _make_boolean_check(
                rank,
                "replicated.sync_names",
                "replicated",
                set(synchronized) == expected_synchronized,
                detail=f"actual={tuple(sorted(synchronized))}, expected={tuple(sorted(expected_synchronized))}",
            )
        )
        _add_replicated_gradient_checks(local_checks, rank, dense, distributed)

        dense_before = _clone_trainable_parameters(dense)
        distributed_before = _clone_trainable_parameters(distributed)
        dense_optimizer.step()
        distributed_optimizer.step()

        local_checks.append(
            _make_boolean_check(
                rank,
                "optimizer.dense_parameters_changed",
                "optimizer",
                _parameters_changed(dense_before, dense),
                detail="at least one dense trainable parameter changed after optimizer step",
            )
        )
        local_checks.append(
            _make_boolean_check(
                rank,
                "optimizer.distributed_parameters_changed",
                "optimizer",
                _parameters_changed(distributed_before, distributed),
                detail="at least one local distributed trainable parameter changed after optimizer step",
            )
        )
        local_checks.append(
            _make_boolean_check(
                rank,
                "optimizer.parameters_finite_after_step",
                "optimizer",
                _trainable_parameters_are_finite(dense) and _trainable_parameters_are_finite(distributed),
                detail="dense and local distributed trainable parameters are finite after SGD step",
            )
        )
        _add_sharded_parameter_checks(local_checks, rank, dense, distributed)
        _add_replicated_parameter_checks(local_checks, rank, dense, distributed)
        _add_optimizer_state_checks(
            local_checks,
            rank,
            optimizer_name,
            dense_optimizer,
            distributed_optimizer,
            dense,
            distributed,
        )
        _tag_step_checks(local_checks, step)

        gathered_step_checks = _gather_checks(local_checks)
        all_checks.extend(gathered_step_checks)
        step_summaries.append(
            _summarize_step_checks(step, gathered_step_checks, dense_loss, distributed_loss)
        )

    failed_checks = [check for check in all_checks if not bool(check["passed"])]
    tensor_checks = [check for check in all_checks if check["max_abs_error"] is not None]
    max_abs_error = max((float(check["max_abs_error"]) for check in tensor_checks), default=0.0)
    first_input_ids, _ = _make_synthetic_training_batch(dense_config, device, 0)

    return {
        "preset": preset,
        "seed": seed,
        "world_size": world_size,
        "vocab_size": dense_config.vocab_size,
        "block_size": dense_config.block_size,
        "n_layer": dense_config.n_layer,
        "n_head": dense_config.n_head,
        "n_embd": dense_config.n_embd,
        "batch_shape": tuple(first_input_ids.shape),
        "steps": steps,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
        "atol": _ATOL,
        "rtol": _RTOL,
        "checks": all_checks,
        "failed_checks": failed_checks,
        "step_summaries": step_summaries,
        "max_abs_error": max_abs_error,
    }


def _validate_optimizer_args(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if optimizer_name not in {"sgd", "adamw"}:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")
    if len(betas) != 2:
        raise ValueError(f"AdamW betas must contain exactly two values, got {betas}")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"AdamW betas must be in [0.0, 1.0), got {betas}")
    if eps <= 0.0:
        raise ValueError(f"AdamW eps must be positive, got {eps}")


def _make_optimizers(
    dense: GPTModel,
    distributed: DistributedGPTModel,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    if optimizer_name == "sgd":
        return (
            torch.optim.SGD(dense.parameters(), lr=learning_rate, weight_decay=weight_decay),
            torch.optim.SGD(distributed.parameters(), lr=learning_rate, weight_decay=weight_decay),
        )
    if optimizer_name == "adamw":
        return (
            torch.optim.AdamW(
                dense.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            ),
            torch.optim.AdamW(
                distributed.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            ),
        )
    raise ValueError(f"unknown optimizer {optimizer_name!r}")


def _add_optimizer_state_checks(
    checks: list[dict[str, object]],
    rank: int,
    optimizer_name: str,
    dense_optimizer: torch.optim.Optimizer,
    distributed_optimizer: torch.optim.Optimizer,
    dense: GPTModel,
    distributed: DistributedGPTModel,
) -> None:
    if optimizer_name != "adamw":
        return
    _add_sharded_optimizer_state_checks(checks, rank, dense_optimizer, distributed_optimizer, dense, distributed)
    _add_replicated_optimizer_state_checks(checks, rank, dense_optimizer, distributed_optimizer, dense, distributed)


def _add_sharded_optimizer_state_checks(
    checks: list[dict[str, object]],
    rank: int,
    dense_optimizer: torch.optim.Optimizer,
    distributed_optimizer: torch.optim.Optimizer,
    dense: GPTModel,
    distributed: DistributedGPTModel,
) -> None:
    start = distributed.token_embedding.local_vocab_start
    end = distributed.token_embedding.local_vocab_end
    _append_optimizer_state_pair(
        checks,
        rank,
        "token_embedding_lm_head.weight_shard",
        "optimizer_state_sharded",
        dense_optimizer,
        distributed_optimizer,
        dense.token_embedding.weight,
        distributed.token_embedding.weight_shards[0],
        dense_slice=lambda tensor: tensor[start:end],
    )

    for index, (dense_block, distributed_block) in enumerate(zip(dense.blocks, distributed.blocks)):
        prefix = f"blocks.{index}"
        qkv = distributed_block.attn.qkv
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.attn.qkv.weight",
            "optimizer_state_sharded",
            dense_optimizer,
            distributed_optimizer,
            dense_block.attn.qkv.weight,
            distributed_block.attn.qkv.weight,
            dense_slice=lambda tensor, qkv=qkv: _local_qkv_slice(tensor, qkv),
        )
        if dense_block.attn.qkv.bias is not None and distributed_block.attn.qkv.bias is not None:
            _append_optimizer_state_pair(
                checks,
                rank,
                f"{prefix}.attn.qkv.bias",
                "optimizer_state_sharded",
                dense_optimizer,
                distributed_optimizer,
                dense_block.attn.qkv.bias,
                distributed_block.attn.qkv.bias,
                dense_slice=lambda tensor, qkv=qkv: _local_qkv_slice(tensor, qkv),
            )

        attn_proj = distributed_block.attn.proj
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.attn.proj.weight",
            "optimizer_state_sharded",
            dense_optimizer,
            distributed_optimizer,
            dense_block.attn.proj.weight,
            distributed_block.attn.proj.weight,
            dense_slice=lambda tensor, module=attn_proj: tensor[:, module.local_in_start : module.local_in_end],
        )

        dense_fc1 = dense_block.mlp.net[0]
        dense_fc2 = dense_block.mlp.net[2]
        fc1 = distributed_block.fc1
        fc2 = distributed_block.fc2
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.mlp.fc1.weight",
            "optimizer_state_sharded",
            dense_optimizer,
            distributed_optimizer,
            dense_fc1.weight,
            distributed_block.fc1.weight,
            dense_slice=lambda tensor, module=fc1: tensor[module.local_out_start : module.local_out_end],
        )
        if dense_fc1.bias is not None and distributed_block.fc1.bias is not None:
            _append_optimizer_state_pair(
                checks,
                rank,
                f"{prefix}.mlp.fc1.bias",
                "optimizer_state_sharded",
                dense_optimizer,
                distributed_optimizer,
                dense_fc1.bias,
                distributed_block.fc1.bias,
                dense_slice=lambda tensor, module=fc1: tensor[module.local_out_start : module.local_out_end],
            )
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.mlp.fc2.weight",
            "optimizer_state_sharded",
            dense_optimizer,
            distributed_optimizer,
            dense_fc2.weight,
            distributed_block.fc2.weight,
            dense_slice=lambda tensor, module=fc2: tensor[:, module.local_in_start : module.local_in_end],
        )


def _add_replicated_optimizer_state_checks(
    checks: list[dict[str, object]],
    rank: int,
    dense_optimizer: torch.optim.Optimizer,
    distributed_optimizer: torch.optim.Optimizer,
    dense: GPTModel,
    distributed: DistributedGPTModel,
) -> None:
    _append_optimizer_state_pair(
        checks,
        rank,
        "position_embedding.weight",
        "optimizer_state_replicated",
        dense_optimizer,
        distributed_optimizer,
        dense.position_embedding.weight,
        distributed.position_embedding.weight,
    )

    for index, (dense_block, distributed_block) in enumerate(zip(dense.blocks, distributed.blocks)):
        prefix = f"blocks.{index}"
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.ln_1.weight",
            "optimizer_state_replicated",
            dense_optimizer,
            distributed_optimizer,
            dense_block.ln_1.weight,
            distributed_block.ln_1.weight,
        )
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.ln_1.bias",
            "optimizer_state_replicated",
            dense_optimizer,
            distributed_optimizer,
            dense_block.ln_1.bias,
            distributed_block.ln_1.bias,
        )
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.ln_2.weight",
            "optimizer_state_replicated",
            dense_optimizer,
            distributed_optimizer,
            dense_block.ln_2.weight,
            distributed_block.ln_2.weight,
        )
        _append_optimizer_state_pair(
            checks,
            rank,
            f"{prefix}.ln_2.bias",
            "optimizer_state_replicated",
            dense_optimizer,
            distributed_optimizer,
            dense_block.ln_2.bias,
            distributed_block.ln_2.bias,
        )
        if dense_block.attn.proj.bias is not None and distributed_block.attn.proj.bias is not None:
            _append_optimizer_state_pair(
                checks,
                rank,
                f"{prefix}.attn.proj.bias",
                "optimizer_state_replicated",
                dense_optimizer,
                distributed_optimizer,
                dense_block.attn.proj.bias,
                distributed_block.attn.proj.bias,
            )
        dense_fc2 = dense_block.mlp.net[2]
        if dense_fc2.bias is not None and distributed_block.fc2.bias is not None:
            _append_optimizer_state_pair(
                checks,
                rank,
                f"{prefix}.mlp.fc2.bias",
                "optimizer_state_replicated",
                dense_optimizer,
                distributed_optimizer,
                dense_fc2.bias,
                distributed_block.fc2.bias,
            )

    _append_optimizer_state_pair(
        checks,
        rank,
        "ln_f.weight",
        "optimizer_state_replicated",
        dense_optimizer,
        distributed_optimizer,
        dense.ln_f.weight,
        distributed.ln_f.weight,
    )
    _append_optimizer_state_pair(
        checks,
        rank,
        "ln_f.bias",
        "optimizer_state_replicated",
        dense_optimizer,
        distributed_optimizer,
        dense.ln_f.bias,
        distributed.ln_f.bias,
    )


def _append_optimizer_state_pair(
    checks: list[dict[str, object]],
    rank: int,
    name: str,
    category: str,
    dense_optimizer: torch.optim.Optimizer,
    distributed_optimizer: torch.optim.Optimizer,
    dense_parameter: torch.nn.Parameter,
    distributed_parameter: torch.nn.Parameter,
    dense_slice: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    dense_state = _require_optimizer_state(dense_optimizer, dense_parameter, f"dense {name}")
    distributed_state = _require_optimizer_state(
        distributed_optimizer,
        distributed_parameter,
        f"distributed {name}",
    )
    for field in ("step", "exp_avg", "exp_avg_sq"):
        expected = _optimizer_state_tensor(dense_state, field, dense_parameter)
        actual = _optimizer_state_tensor(distributed_state, field, distributed_parameter)
        if dense_slice is not None and field != "step":
            expected = dense_slice(expected)
        checks.append(
            _make_tensor_check(
                rank,
                f"{name}.adamw.{field}",
                category,
                expected,
                actual,
            )
        )


def _require_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
    name: str,
) -> dict[str, object]:
    state = optimizer.state.get(parameter)
    if not state:
        raise AssertionError(f"{name} has no optimizer state")
    return state


def _optimizer_state_tensor(
    state: dict[str, object],
    field: str,
    reference: torch.nn.Parameter,
) -> torch.Tensor:
    if field not in state:
        raise AssertionError(f"optimizer state is missing {field!r}")
    value = state[field]
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.tensor(value, device=reference.device)
    if field == "step":
        return tensor.to(dtype=torch.float32)
    return tensor


def _make_synthetic_training_batch(
    config: object,
    device: torch.device,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(8, int(getattr(config, "block_size")))
    batch_size = 2
    vocab_size = int(getattr(config, "vocab_size"))
    base = torch.arange(batch_size * seq_len, dtype=torch.long, device=device).view(batch_size, seq_len)
    input_ids = (base * 7 + 3 + step * 11) % vocab_size
    targets = (input_ids + 5 + step * 3) % vocab_size
    return input_ids, targets


def _clone_trainable_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]


def _parameters_changed(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    after = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
    if len(before) != len(after):
        return False
    return bool(any(not torch.equal(old, new) for old, new in zip(before, after)))


def _trainable_parameters_are_finite(model: torch.nn.Module) -> bool:
    parameters = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
    return bool(parameters and all(torch.isfinite(parameter).all() for parameter in parameters))


def _tag_step_checks(checks: list[dict[str, object]], step: int) -> None:
    for check in checks:
        check["step"] = step
        check["name"] = f"step.{step:03d}.{check['name']}"


def _summarize_step_checks(
    step: int,
    checks: list[dict[str, object]],
    dense_loss: torch.Tensor,
    distributed_loss: torch.Tensor,
) -> dict[str, object]:
    failed = [check for check in checks if not bool(check["passed"])]
    tensor_checks = [check for check in checks if check["max_abs_error"] is not None]
    max_abs_error = max((float(check["max_abs_error"]) for check in tensor_checks), default=0.0)
    return {
        "step": step,
        "checks": len(checks),
        "failed": len(failed),
        "dense_loss": float(dense_loss.detach().item()),
        "distributed_loss": float(distributed_loss.detach().item()),
        "max_abs_error": max_abs_error,
    }


def _print_report(
    result: dict[str, object],
    backend: str,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    checks = list(result["checks"])
    failed_checks = list(result["failed_checks"])
    tensor_checks = [check for check in checks if check["max_abs_error"] is not None]
    passed = len(checks) - len(failed_checks)

    print("CUDA/NCCL distributed GPT multi-step training equivalence demo")
    print(f"backend: {backend}")
    print(f"world_size: {result['world_size']}")
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
    print(f"batch shape: {result['batch_shape']}")
    print(f"training steps: {result['steps']}")
    print(f"optimizer: {result['optimizer']}")
    print(f"learning rate: {result['learning_rate']}")
    print(f"weight decay: {result['weight_decay']}")
    if result["optimizer"] == "adamw":
        print(f"AdamW betas: {result['betas']}")
        print(f"AdamW eps: {result['eps']}")
    print(f"tolerance: atol={result['atol']}, rtol={result['rtol']}")
    print()
    print("Training checks")
    print(f"  total checks: {len(checks)}")
    print(f"  passed checks: {passed}")
    print(f"  failed checks: {len(failed_checks)}")
    print(f"  max abs error: {result['max_abs_error']:.6e}")
    print()
    print("Per-step summary")
    for summary in result["step_summaries"]:
        print(
            f"  step {summary['step']}: checks={summary['checks']} failed={summary['failed']} "
            f"dense_loss={summary['dense_loss']:.6e} "
            f"distributed_loss={summary['distributed_loss']:.6e} "
            f"max_abs_error={summary['max_abs_error']:.6e}"
        )

    if failed_checks:
        print()
        print("Failed checks")
        for check in failed_checks[: args.max_report]:
            print(_format_check(check))
        if len(failed_checks) > args.max_report:
            print(f"  ... {len(failed_checks) - args.max_report} more failed checks")

    print()
    print("Largest tensor errors")
    largest = sorted(tensor_checks, key=lambda item: float(item["max_abs_error"]), reverse=True)
    for check in largest[: args.max_report]:
        print(_format_check(check))

    print()
    print("Note:")
    print("  This runs a short optimizer loop and compares every step with dense GPT slices.")
    print("  Each step checks logits, loss, gradient shards, replicated gradients, and updated shards.")
    print("  With --optimizer adamw, it also checks exp_avg, exp_avg_sq, and step state shards.")
    print("  Replicated gradients are synchronized explicitly before each local optimizer step.")
    print("  Dropout is disabled in the presets; CUDA RNG tracking is not claimed here.")
    print("  This is an isolated CUDA/NCCL prototype validation path.")
    print("  Real distributed GPT TP is not wired into the main GPTModel path.")
    print("  No multi-node orchestration or speedup claims.")


def _assert_strict_training_result(result: dict[str, object]) -> None:
    failed_checks = list(result["failed_checks"])
    if not failed_checks:
        return
    preview = ", ".join(f"rank {check['rank']} {check['name']}" for check in failed_checks[:12])
    if len(failed_checks) > 12:
        preview += f", ... {len(failed_checks) - 12} more"
    raise AssertionError(
        "strict CUDA/NCCL distributed GPT multi-step training equivalence failed checks: "
        + preview
        + f"; max_abs_error={result['max_abs_error']}"
    )


if __name__ == "__main__":
    main()
