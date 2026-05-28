"""Forward/backward loop over microbatches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nano_megatron_engine.engine.microbatch import split_batch


@dataclass(frozen=True)
class ForwardBackwardResult:
    loss: float
    tokens: int
    loss_tokens: int
    num_microbatches: int


def run_forward_backward(
    model: nn.Module,
    batch: Any,
    micro_batch_size: int,
) -> ForwardBackwardResult:
    """Accumulate gradients by running forward/backward per microbatch."""

    microbatches = split_batch(batch, micro_batch_size)
    total_loss_tokens = sum(count_loss_tokens(microbatch) for microbatch in microbatches)
    if total_loss_tokens <= 0:
        raise ValueError("batch must contain at least one next-token target")

    weighted_loss = 0.0
    for microbatch in microbatches:
        input_ids, targets = unpack_language_model_batch(microbatch)
        _, loss = model(input_ids, targets=targets)
        if loss is None:
            raise ValueError("model did not return a loss")
        micro_loss_tokens = count_loss_tokens(microbatch)
        scale = micro_loss_tokens / total_loss_tokens
        (loss * scale).backward()
        weighted_loss += float(loss.detach().item()) * scale

    return ForwardBackwardResult(
        loss=weighted_loss,
        tokens=count_tokens(batch),
        loss_tokens=total_loss_tokens,
        num_microbatches=len(microbatches),
    )


def unpack_language_model_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either a token tensor or a dict with input_ids and labels."""

    if torch.is_tensor(batch):
        return batch, batch

    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("language model batch dict must contain input_ids")
        input_ids = batch["input_ids"]
        targets = batch.get("targets", batch.get("labels", input_ids))
        if not torch.is_tensor(input_ids) or not torch.is_tensor(targets):
            raise TypeError("input_ids and targets/labels must be tensors")
        return input_ids, targets

    raise TypeError("language model batch must be a tensor or dict")


def count_tokens(batch: Any) -> int:
    input_ids, _ = unpack_language_model_batch(batch)
    return int(input_ids.numel())


def count_loss_tokens(batch: Any) -> int:
    input_ids, _ = unpack_language_model_batch(batch)
    if input_ids.ndim < 2:
        raise ValueError("language model input_ids must have shape (batch, sequence)")
    return int(input_ids.shape[0] * max(input_ids.shape[1] - 1, 0))

