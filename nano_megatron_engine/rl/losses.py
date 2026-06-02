"""Small PPO-style RL losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from nano_megatron_engine.rl.data import masked_mean


def ppo_clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped PPO policy loss and approximate KL."""

    _validate_same_shape(new_logprobs, old_logprobs, advantages, mask)
    if clip_epsilon < 0.0:
        raise ValueError(f"clip_epsilon must be non-negative, got {clip_epsilon}")

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
    approx_kl = masked_mean(old_logprobs - new_logprobs, mask)
    return loss, approx_kl


def value_loss(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return masked half-MSE value loss."""

    _validate_value_shapes(values, returns, mask)
    return 0.5 * masked_mean((values - returns).square(), mask)


def entropy_bonus(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return masked categorical entropy from logits."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, actions, vocab], got ndim={logits.ndim}")
    if logits.shape[:2] != mask.shape:
        raise ValueError(f"logits batch/action shape and mask shape must match, got {logits.shape[:2]} and {mask.shape}")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return masked_mean(entropy, mask)


def _validate_same_shape(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    for name, tensor in (
        ("new_logprobs", new_logprobs),
        ("old_logprobs", old_logprobs),
        ("advantages", advantages),
        ("mask", mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, actions], got ndim={tensor.ndim}")
    if (
        new_logprobs.shape != old_logprobs.shape
        or new_logprobs.shape != advantages.shape
        or new_logprobs.shape != mask.shape
    ):
        raise ValueError("new_logprobs, old_logprobs, advantages, and mask must all have the same shape")


def _validate_value_shapes(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> None:
    for name, tensor in (("values", values), ("returns", returns), ("mask", mask)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, actions], got ndim={tensor.ndim}")
    if values.shape != returns.shape or values.shape != mask.shape:
        raise ValueError("values, returns, and mask must all have the same shape")
