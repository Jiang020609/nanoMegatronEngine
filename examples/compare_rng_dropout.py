"""Check dropout determinism with the lightweight RNG state tracker."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from nano_megatron_engine.parallel import RNGStateTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RNGStateTracker dropout determinism.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=14001)
    parser.add_argument("--dropout-seed", type=int, default=14002)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    result = _run_check(device=device, seed=args.seed, dropout_seed=args.dropout_seed, dropout=args.dropout)

    print("RNG tracker dropout determinism check")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"dropout seed: {args.dropout_seed}")
    print(f"dropout: {args.dropout}")
    print(f"replay close: {result['replay_close']}")
    print(f"stream advances: {result['stream_advances']}")
    print(f"outer RNG restored: {result['outer_rng_restored']}")
    print(f"replay max abs error: {result['replay_max_abs_error']:.6e}")
    print()
    print("Note:")
    print("  This checks process-local RNG tracker behavior for dropout masks.")
    print("  It is a foundation for later TP dropout/RNG work.")
    print("  It is not yet tensor-parallel rank RNG integration.")

    if not all(bool(result[name]) for name in ("replay_close", "stream_advances", "outer_rng_restored")):
        raise AssertionError(f"RNG tracker dropout determinism check failed: {result}")


def _run_check(
    device: torch.device,
    seed: int,
    dropout_seed: int,
    dropout: float,
) -> dict[str, object]:
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.arange(32, dtype=torch.float32, device=device).view(4, 8)

    tracker = RNGStateTracker(device=device)
    tracker.add("dropout", dropout_seed)
    initial_dropout_state = tracker.get_state("dropout")

    with tracker.fork("dropout"):
        first = F.dropout(x, p=dropout, training=True)
    with tracker.fork("dropout"):
        second = F.dropout(x, p=dropout, training=True)

    tracker.set_state("dropout", initial_dropout_state)
    with tracker.fork("dropout"):
        replay = F.dropout(x, p=dropout, training=True)

    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
    expected_outer_before = torch.rand(3, device=device)
    expected_outer_after = torch.rand(3, device=device)

    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
    actual_outer_before = torch.rand(3, device=device)
    with tracker.fork("dropout"):
        _ = F.dropout(x, p=dropout, training=True)
    actual_outer_after = torch.rand(3, device=device)

    replay_close = bool(torch.allclose(first, replay))
    replay_error = float((first - replay).abs().max().item())
    stream_advances = bool(not torch.equal(first, second))
    outer_rng_restored = bool(
        torch.allclose(actual_outer_before, expected_outer_before)
        and torch.allclose(actual_outer_after, expected_outer_after)
    )

    return {
        "replay_close": replay_close,
        "stream_advances": stream_advances,
        "outer_rng_restored": outer_rng_restored,
        "replay_max_abs_error": replay_error,
    }


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requires torch.cuda.is_available()")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


if __name__ == "__main__":
    main()
