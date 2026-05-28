"""Benchmark tiny GPT microbatch training."""

from __future__ import annotations

import argparse

from nano_megatron_engine.benchmark import run_microbatch_benchmark
from nano_megatron_engine.memory.estimator import format_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small microbatch throughput benchmark.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()

    result = run_microbatch_benchmark(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        steps=args.steps,
    )

    print(f"device: {result.device}")
    print(f"average step time: {result.average_step_time_sec:.4f} sec")
    print(f"tokens/sec: {result.tokens_per_sec:.1f}")
    print(f"final loss: {result.final_loss:.4f}")
    if result.peak_cuda_memory_bytes is None:
        print("peak CUDA memory: unavailable on CPU")
    else:
        print(f"peak CUDA memory: {format_bytes(result.peak_cuda_memory_bytes)}")


if __name__ == "__main__":
    main()

