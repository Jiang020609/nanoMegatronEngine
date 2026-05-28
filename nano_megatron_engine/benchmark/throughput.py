"""Throughput benchmark for tiny GPT microbatch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nano_megatron_engine.engine import Trainer
from nano_megatron_engine.model import GPTConfig, GPTModel
from nano_megatron_engine.utils.seed import set_seed


@dataclass(frozen=True)
class BenchmarkResult:
    average_step_time_sec: float
    tokens_per_sec: float
    final_loss: float
    peak_cuda_memory_bytes: int | None
    device: str


def run_microbatch_benchmark(
    config: GPTConfig | None = None,
    batch_size: int = 8,
    micro_batch_size: int = 2,
    sequence_length: int = 32,
    steps: int = 5,
    device: str | None = None,
) -> BenchmarkResult:
    """Run a small training benchmark and return aggregate metrics."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    set_seed(1234)
    config = config or GPTConfig(vocab_size=256, block_size=sequence_length, n_layer=2, n_head=2, n_embd=64)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GPTModel(config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = Trainer(model, optimizer, micro_batch_size=micro_batch_size, device=selected_device)

    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)

    total_time = 0.0
    total_tokens = 0
    final_loss = 0.0
    for _ in range(steps):
        batch = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, sequence_length),
            device=selected_device,
        )
        result = trainer.train_step(batch)
        total_time += result.step_time_sec
        total_tokens += result.tokens
        final_loss = result.loss

    peak_memory = torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else None
    return BenchmarkResult(
        average_step_time_sec=total_time / steps,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else float("inf"),
        final_loss=final_loss,
        peak_cuda_memory_bytes=peak_memory,
        device=str(selected_device),
    )

