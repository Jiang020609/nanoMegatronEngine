"""Run a CUDA/NCCL distributed GPT gradient-slice equivalence comparison."""

from __future__ import annotations

import argparse
import os

import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import get_backend, get_rank, get_world_size, init_distributed_from_env

_ATOL = 1e-5
_RTOL = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense GPT gradients against CUDA/NCCL distributed GPT local gradient shards."
    )
    parser.add_argument(
        "--preset",
        choices=("tiny", "small"),
        default="tiny",
        help="model shape preset; small is still a smoke test but exercises wider tensors",
    )
    parser.add_argument("--seed", type=int, default=13201, help="deterministic seed for model initialization")
    parser.add_argument("--max-report", type=int, default=24, help="maximum failed/largest-error checks to print")
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="print gradient checks without failing on mismatches",
    )
    parser.set_defaults(strict=True)
    args = parser.parse_args()

    if not _has_torchrun_env():
        print("CUDA/NCCL distributed GPT gradient equivalence demo")
        print("Run with torchrun on a CUDA PyTorch build with NCCL, for example:")
        print("  torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_gradients_nccl.py")
        print(
            "  torchrun --standalone --nproc_per_node=4 "
            "examples/compare_distributed_gpt_gradients_nccl.py --preset small"
        )
        print("Strict validation is enabled by default; use --no-strict to print without failing.")
        print("This compares local distributed gradient shards with dense GPT gradient slices.")
        print("The main GPTModel path is not wired to real distributed TP.")
        print("No multi-node orchestration or speedup claims.")
        return

    _run_demo(args)


def _run_demo(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/NCCL distributed GPT gradient equivalence requires torch.cuda.is_available()")
    if not dist.is_available() or not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available():
        raise RuntimeError("CUDA/NCCL distributed GPT gradient equivalence requires a PyTorch build with NCCL support")

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
                "this CUDA/NCCL gradient equivalence demo expects world_size to divide "
                f"n_head={dense_config.n_head} and vocab_size={dense_config.vocab_size}; "
                "try --nproc_per_node=2 or --nproc_per_node=4 with the default presets"
            )

        result = _compare_gradient_equivalence(world_size, device, preset=args.preset, seed=args.seed)
        torch.cuda.synchronize(device)

        if rank == 0:
            _print_report(result, backend=backend, device=device, args=args)

        _synchronize_and_barrier(device)
        if args.strict:
            _assert_strict_result(result)
        _synchronize_and_barrier(device)
        completed = True
    finally:
        if dist.is_available() and dist.is_initialized():
            if completed:
                torch.cuda.synchronize(device)
            dist.destroy_process_group()


def _compare_gradient_equivalence(
    world_size: int,
    device: torch.device,
    preset: str,
    seed: int,
) -> dict[str, object]:
    _make_cuda_math_deterministic_enough()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dense_config = _dense_config(preset)
    distributed_config = _distributed_config(world_size, preset)
    dense = GPTModel(dense_config).to(device)
    dense.eval()
    distributed = DistributedGPTModel(distributed_config).to(device)
    distributed.eval()
    distributed.copy_from_dense_(dense)

    input_ids, targets = _make_synthetic_batch(dense_config, device)

    dense.zero_grad(set_to_none=True)
    distributed.zero_grad(set_to_none=True)
    dense_logits, dense_loss = dense(input_ids, targets)
    distributed_logits, distributed_loss = distributed(input_ids, targets)
    assert dense_loss is not None
    assert distributed_loss is not None
    dense_loss.backward()
    distributed_loss.backward()

    rank = get_rank()
    local_checks: list[dict[str, object]] = [
        _make_tensor_check(rank, "forward.logits", "forward", dense_logits, distributed_logits),
        _make_tensor_check(rank, "forward.loss", "forward", dense_loss, distributed_loss),
    ]
    _add_sharded_gradient_checks(local_checks, rank, dense, distributed)

    synchronized = distributed.synchronize_replicated_gradients_()
    expected_synchronized = set(distributed.replicated_parameter_names())
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

    all_checks = _gather_checks(local_checks)
    failed_checks = [check for check in all_checks if not bool(check["passed"])]
    tensor_checks = [check for check in all_checks if check["max_abs_error"] is not None]
    max_abs_error = max((float(check["max_abs_error"]) for check in tensor_checks), default=0.0)

    return {
        "preset": preset,
        "seed": seed,
        "world_size": world_size,
        "vocab_size": dense_config.vocab_size,
        "block_size": dense_config.block_size,
        "n_layer": dense_config.n_layer,
        "n_head": dense_config.n_head,
        "n_embd": dense_config.n_embd,
        "batch_shape": tuple(input_ids.shape),
        "atol": _ATOL,
        "rtol": _RTOL,
        "checks": all_checks,
        "failed_checks": failed_checks,
        "max_abs_error": max_abs_error,
    }


def _add_sharded_gradient_checks(
    checks: list[dict[str, object]],
    rank: int,
    dense: GPTModel,
    distributed: DistributedGPTModel,
) -> None:
    start = distributed.token_embedding.local_vocab_start
    end = distributed.token_embedding.local_vocab_end
    checks.append(
        _make_tensor_check(
            rank,
            "token_embedding_lm_head.weight_shard",
            "sharded",
            _require_grad(dense.token_embedding.weight, "dense token_embedding.weight")[start:end],
            _require_grad(distributed.token_embedding.weight_shards[0], "distributed token_embedding.weight_shards.0"),
        )
    )

    for index, (dense_block, distributed_block) in enumerate(zip(dense.blocks, distributed.blocks)):
        prefix = f"blocks.{index}"
        qkv = distributed_block.attn.qkv
        checks.append(
            _make_tensor_check(
                rank,
                f"{prefix}.attn.qkv.weight",
                "sharded",
                _local_qkv_slice(_require_grad(dense_block.attn.qkv.weight, f"dense {prefix}.attn.qkv.weight"), qkv),
                _require_grad(distributed_block.attn.qkv.weight, f"distributed {prefix}.attn.qkv.weight"),
            )
        )
        if dense_block.attn.qkv.bias is not None and distributed_block.attn.qkv.bias is not None:
            checks.append(
                _make_tensor_check(
                    rank,
                    f"{prefix}.attn.qkv.bias",
                    "sharded",
                    _local_qkv_slice(_require_grad(dense_block.attn.qkv.bias, f"dense {prefix}.attn.qkv.bias"), qkv),
                    _require_grad(distributed_block.attn.qkv.bias, f"distributed {prefix}.attn.qkv.bias"),
                )
            )

        attn_proj = distributed_block.attn.proj
        checks.append(
            _make_tensor_check(
                rank,
                f"{prefix}.attn.proj.weight",
                "sharded",
                _require_grad(dense_block.attn.proj.weight, f"dense {prefix}.attn.proj.weight")[
                    :, attn_proj.local_in_start : attn_proj.local_in_end
                ],
                _require_grad(distributed_block.attn.proj.weight, f"distributed {prefix}.attn.proj.weight"),
            )
        )

        dense_fc1 = dense_block.mlp.net[0]
        dense_fc2 = dense_block.mlp.net[2]
        fc1 = distributed_block.fc1
        fc2 = distributed_block.fc2
        checks.append(
            _make_tensor_check(
                rank,
                f"{prefix}.mlp.fc1.weight",
                "sharded",
                _require_grad(dense_fc1.weight, f"dense {prefix}.mlp.fc1.weight")[
                    fc1.local_out_start : fc1.local_out_end
                ],
                _require_grad(distributed_block.fc1.weight, f"distributed {prefix}.mlp.fc1.weight"),
            )
        )
        if dense_fc1.bias is not None and distributed_block.fc1.bias is not None:
            checks.append(
                _make_tensor_check(
                    rank,
                    f"{prefix}.mlp.fc1.bias",
                    "sharded",
                    _require_grad(dense_fc1.bias, f"dense {prefix}.mlp.fc1.bias")[
                        fc1.local_out_start : fc1.local_out_end
                    ],
                    _require_grad(distributed_block.fc1.bias, f"distributed {prefix}.mlp.fc1.bias"),
                )
            )
        checks.append(
            _make_tensor_check(
                rank,
                f"{prefix}.mlp.fc2.weight",
                "sharded",
                _require_grad(dense_fc2.weight, f"dense {prefix}.mlp.fc2.weight")[
                    :, fc2.local_in_start : fc2.local_in_end
                ],
                _require_grad(distributed_block.fc2.weight, f"distributed {prefix}.mlp.fc2.weight"),
            )
        )


def _add_replicated_gradient_checks(
    checks: list[dict[str, object]],
    rank: int,
    dense: GPTModel,
    distributed: DistributedGPTModel,
) -> None:
    checks.append(
        _make_tensor_check(
            rank,
            "position_embedding.weight",
            "replicated",
            _require_grad(dense.position_embedding.weight, "dense position_embedding.weight"),
            _require_grad(distributed.position_embedding.weight, "distributed position_embedding.weight"),
        )
    )

    for index, (dense_block, distributed_block) in enumerate(zip(dense.blocks, distributed.blocks)):
        prefix = f"blocks.{index}"
        _append_replicated_pair(checks, rank, f"{prefix}.ln_1.weight", dense_block.ln_1.weight, distributed_block.ln_1.weight)
        _append_replicated_pair(checks, rank, f"{prefix}.ln_1.bias", dense_block.ln_1.bias, distributed_block.ln_1.bias)
        _append_replicated_pair(checks, rank, f"{prefix}.ln_2.weight", dense_block.ln_2.weight, distributed_block.ln_2.weight)
        _append_replicated_pair(checks, rank, f"{prefix}.ln_2.bias", dense_block.ln_2.bias, distributed_block.ln_2.bias)
        if dense_block.attn.proj.bias is not None and distributed_block.attn.proj.bias is not None:
            _append_replicated_pair(
                checks,
                rank,
                f"{prefix}.attn.proj.bias",
                dense_block.attn.proj.bias,
                distributed_block.attn.proj.bias,
            )
        dense_fc2 = dense_block.mlp.net[2]
        if dense_fc2.bias is not None and distributed_block.fc2.bias is not None:
            _append_replicated_pair(checks, rank, f"{prefix}.mlp.fc2.bias", dense_fc2.bias, distributed_block.fc2.bias)

    _append_replicated_pair(checks, rank, "ln_f.weight", dense.ln_f.weight, distributed.ln_f.weight)
    _append_replicated_pair(checks, rank, "ln_f.bias", dense.ln_f.bias, distributed.ln_f.bias)


def _append_replicated_pair(
    checks: list[dict[str, object]],
    rank: int,
    name: str,
    dense_parameter: torch.nn.Parameter,
    distributed_parameter: torch.nn.Parameter,
) -> None:
    checks.append(
        _make_tensor_check(
            rank,
            name,
            "replicated",
            _require_grad(dense_parameter, f"dense {name}"),
            _require_grad(distributed_parameter, f"distributed {name}"),
        )
    )


def _local_qkv_slice(dense_tensor: torch.Tensor, qkv_module: object) -> torch.Tensor:
    hidden_size = int(getattr(qkv_module, "hidden_size"))
    local_start = int(getattr(qkv_module, "local_start"))
    local_end = int(getattr(qkv_module, "local_end"))
    return torch.cat(
        [
            dense_tensor[local_start:local_end],
            dense_tensor[hidden_size + local_start : hidden_size + local_end],
            dense_tensor[2 * hidden_size + local_start : 2 * hidden_size + local_end],
        ],
        dim=0,
    )


def _require_grad(parameter: torch.nn.Parameter, name: str) -> torch.Tensor:
    if parameter.grad is None:
        raise AssertionError(f"{name} has no gradient")
    return parameter.grad


def _make_tensor_check(
    rank: int,
    name: str,
    category: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, object]:
    expected = expected.detach()
    actual = actual.detach()
    expected_shape = tuple(expected.shape)
    actual_shape = tuple(actual.shape)
    shape_matches = expected_shape == actual_shape
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    if shape_matches:
        max_abs_error = float((expected - actual).abs().max().item()) if expected.numel() > 0 else 0.0
        close = bool(torch.allclose(actual, expected, atol=_ATOL, rtol=_RTOL))
    else:
        max_abs_error = None
        close = False
    return {
        "rank": rank,
        "name": name,
        "category": category,
        "passed": bool(shape_matches and finite and close),
        "shape_matches": shape_matches,
        "finite": finite,
        "close": close,
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "max_abs_error": max_abs_error,
        "expected_norm": float(expected.norm().item()) if expected.numel() > 0 else 0.0,
        "actual_norm": float(actual.norm().item()) if actual.numel() > 0 else 0.0,
        "detail": "",
    }


def _make_boolean_check(
    rank: int,
    name: str,
    category: str,
    passed: bool,
    detail: str,
) -> dict[str, object]:
    return {
        "rank": rank,
        "name": name,
        "category": category,
        "passed": bool(passed),
        "shape_matches": bool(passed),
        "finite": bool(passed),
        "close": bool(passed),
        "expected_shape": None,
        "actual_shape": None,
        "max_abs_error": None,
        "expected_norm": None,
        "actual_norm": None,
        "detail": detail,
    }


def _gather_checks(local_checks: list[dict[str, object]]) -> list[dict[str, object]]:
    import torch.distributed as dist

    gathered: list[list[dict[str, object]] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_checks)
    return [check for rank_checks in gathered if rank_checks is not None for check in rank_checks]


def _dense_config(preset: str) -> GPTConfig:
    if preset == "small":
        return GPTConfig(
            vocab_size=128,
            block_size=16,
            n_layer=2,
            n_head=8,
            n_embd=64,
            dropout=0.0,
            tensor_parallel_size=1,
        )
    if preset != "tiny":
        raise ValueError(f"unknown CUDA/NCCL GPT gradient preset {preset!r}")
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=1,
    )


def _distributed_config(world_size: int, preset: str) -> GPTConfig:
    config = _dense_config(preset)
    return GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.0,
        tensor_parallel_size=world_size,
    )


def _make_synthetic_batch(config: GPTConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(8, config.block_size)
    input_ids = torch.arange(2 * seq_len, dtype=torch.long, device=device).view(2, seq_len)
    input_ids = (input_ids * 7 + 3) % config.vocab_size
    targets = (input_ids + 5) % config.vocab_size
    return input_ids, targets


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

    print("CUDA/NCCL distributed GPT gradient equivalence demo")
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
    print(f"tolerance: atol={result['atol']}, rtol={result['rtol']}")
    print()
    print("Gradient checks")
    print(f"  total checks: {len(checks)}")
    print(f"  passed checks: {passed}")
    print(f"  failed checks: {len(failed_checks)}")
    print(f"  max abs error: {result['max_abs_error']:.6e}")

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
    print("  This compares real dense GPT gradients with local distributed GPT gradient shards.")
    print("  Sharded parameters are compared against the matching dense row/column/head slice.")
    print("  Replicated parameters are compared after explicit replicated-gradient synchronization.")
    print("  This is an isolated CUDA/NCCL prototype validation path.")
    print("  Real distributed GPT TP is not wired into the main GPTModel path.")
    print("  No multi-node orchestration or speedup claims.")


def _format_check(check: dict[str, object]) -> str:
    error = check["max_abs_error"]
    error_text = "n/a" if error is None else f"{float(error):.6e}"
    detail = f" detail={check['detail']}" if check["detail"] else ""
    return (
        f"  rank {check['rank']} {check['category']} {check['name']}: "
        f"passed={check['passed']} close={check['close']} finite={check['finite']} "
        f"expected_shape={check['expected_shape']} actual_shape={check['actual_shape']} "
        f"max_abs_error={error_text}{detail}"
    )


def _assert_strict_result(result: dict[str, object]) -> None:
    failed_checks = list(result["failed_checks"])
    if not failed_checks:
        return
    preview = ", ".join(f"rank {check['rank']} {check['name']}" for check in failed_checks[:12])
    if len(failed_checks) > 12:
        preview += f", ... {len(failed_checks) - 12} more"
    raise AssertionError(
        "strict CUDA/NCCL distributed GPT gradient equivalence failed checks: "
        + preview
        + f"; max_abs_error={result['max_abs_error']}"
    )


def _synchronize_and_barrier(device: torch.device) -> None:
    import torch.distributed as dist

    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)


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
