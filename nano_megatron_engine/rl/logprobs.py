"""Token log-probability helpers for RL objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from nano_megatron_engine.rl.data import masked_mean


def next_token_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Gather log-probs of ``tokens[:, 1:]`` from ``logits[:, :-1]``."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got ndim={logits.ndim}")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [batch, sequence], got ndim={tokens.ndim}")
    if logits.shape[:2] != tokens.shape:
        raise ValueError(f"logits and tokens batch/sequence shapes must match, got {logits.shape[:2]} and {tokens.shape}")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2 to gather next-token log-probs")
    if tokens.device != logits.device:
        raise ValueError("tokens must be on the same device as logits")

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens = tokens[:, 1:].unsqueeze(-1)
    return log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)


def masked_kl_divergence(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean token KL estimate ``policy_logprobs - reference_logprobs`` over a mask."""

    if policy_logprobs.shape != reference_logprobs.shape:
        raise ValueError(
            "policy_logprobs and reference_logprobs must have the same shape, "
            f"got {policy_logprobs.shape} and {reference_logprobs.shape}"
        )
    return masked_mean(policy_logprobs - reference_logprobs, mask)
