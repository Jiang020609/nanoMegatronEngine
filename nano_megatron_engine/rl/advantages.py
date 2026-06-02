"""Advantage and return estimators."""

from __future__ import annotations

import torch


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked generalized advantage estimates and returns."""

    _validate_shapes(rewards, values, next_values, mask)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0.0, 1.0], got {lam}")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    float_mask = mask.to(dtype=rewards.dtype)
    for index in reversed(range(rewards.shape[1])):
        active = float_mask[:, index]
        delta = rewards[:, index] + gamma * next_values[:, index] * active - values[:, index]
        next_advantage = (delta + gamma * lam * active * next_advantage) * active
        advantages[:, index] = next_advantage

    returns = (advantages + values) * float_mask
    return advantages, returns


def _validate_shapes(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    for name, tensor in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
        ("mask", mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, actions], got ndim={tensor.ndim}")
    if rewards.shape != values.shape or rewards.shape != next_values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values, next_values, and mask must all have the same shape")
