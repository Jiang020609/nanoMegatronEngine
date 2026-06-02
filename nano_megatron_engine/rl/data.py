"""RL batch masking helpers."""

from __future__ import annotations

import torch


def response_action_mask(tokens: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
    """Return a next-token action mask for response tokens."""

    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
    if not isinstance(prompt_lengths, torch.Tensor):
        raise TypeError(f"prompt_lengths must be a torch.Tensor, got {type(prompt_lengths).__name__}")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [batch, sequence], got ndim={tokens.ndim}")
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != tokens.shape[0]:
        raise ValueError(
            "prompt_lengths must have shape [batch] matching tokens, "
            f"got prompt_lengths={tuple(prompt_lengths.shape)}, tokens={tuple(tokens.shape)}"
        )
    if tokens.shape[1] < 2:
        raise ValueError("tokens sequence length must be at least 2 to build next-token action masks")
    if prompt_lengths.device != tokens.device:
        raise ValueError("prompt_lengths must be on the same device as tokens")
    if torch.any(prompt_lengths < 1) or torch.any(prompt_lengths >= tokens.shape[1]):
        raise ValueError(
            "prompt_lengths must be in [1, sequence_length - 1] so each row has at least one response token"
        )

    target_positions = torch.arange(1, tokens.shape[1], device=tokens.device).unsqueeze(0)
    return target_positions >= prompt_lengths.to(device=tokens.device).unsqueeze(1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean of ``values`` over true mask entries."""

    _validate_same_shape(values, mask, "values", "mask")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    float_mask = mask.to(dtype=values.dtype)
    denom = float_mask.sum().clamp_min(eps)
    return (values * float_mask).sum() / denom


def _validate_same_shape(left: torch.Tensor, right: torch.Tensor, left_name: str, right_name: str) -> None:
    if not isinstance(left, torch.Tensor):
        raise TypeError(f"{left_name} must be a torch.Tensor, got {type(left).__name__}")
    if not isinstance(right, torch.Tensor):
        raise TypeError(f"{right_name} must be a torch.Tensor, got {type(right).__name__}")
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape, got {left.shape} and {right.shape}")
