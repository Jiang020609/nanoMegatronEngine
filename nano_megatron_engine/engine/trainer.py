"""A tiny single-device trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nano_megatron_engine.engine.forward_backward import run_forward_backward


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    tokens: int
    tokens_per_sec: float
    step_time_sec: float
    num_microbatches: int


class Trainer:
    """Simple gradient-accumulating trainer for one CPU or CUDA device."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        micro_batch_size: int,
        grad_clip_norm: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")

        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.micro_batch_size = micro_batch_size
        self.grad_clip_norm = grad_clip_norm
        self.loss_history: list[float] = []

    def train_step(self, batch: Any) -> TrainStepResult:
        """Run one optimizer step over all microbatches."""

        self.model.train()
        batch = move_to_device(batch, self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        result = run_forward_backward(self.model, batch, self.micro_batch_size)

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start

        tokens_per_sec = result.tokens / elapsed if elapsed > 0 else float("inf")
        self.loss_history.append(result.loss)
        return TrainStepResult(
            loss=result.loss,
            tokens=result.tokens,
            tokens_per_sec=tokens_per_sec,
            step_time_sec=elapsed,
            num_microbatches=result.num_microbatches,
        )


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch

