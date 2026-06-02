"""Reward shaping helpers."""

from __future__ import annotations

import torch


def terminal_reward_to_token_rewards(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Place one scalar score on each row's final active action."""

    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"scores must be a torch.Tensor, got {type(scores).__name__}")
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if scores.ndim != 1:
        raise ValueError(f"scores must have shape [batch], got ndim={scores.ndim}")
    if mask.ndim != 2 or mask.shape[0] != scores.shape[0]:
        raise ValueError(f"mask must have shape [batch, actions], got {tuple(mask.shape)}")
    if scores.device != mask.device:
        raise ValueError("scores and mask must be on the same device")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each row in mask must contain at least one active action")

    rewards = torch.zeros(mask.shape, dtype=scores.dtype, device=scores.device)
    active_positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    last_indices = (active_positions * mask.to(dtype=torch.long)).max(dim=1).values
    final_positions = active_positions == last_indices.unsqueeze(1)
    rewards[final_positions & mask] = scores
    return rewards


def apply_kl_penalty(
    token_rewards: torch.Tensor,
    kl: torch.Tensor,
    mask: torch.Tensor,
    *,
    kl_coef: float,
) -> torch.Tensor:
    """Subtract a token-level KL penalty from active rewards."""

    if kl_coef < 0.0:
        raise ValueError(f"kl_coef must be non-negative, got {kl_coef}")
    if token_rewards.shape != kl.shape or token_rewards.shape != mask.shape:
        raise ValueError("token_rewards, kl, and mask must all have the same shape")
    return (token_rewards - kl_coef * kl) * mask.to(dtype=token_rewards.dtype)
