"""Activation checkpointing wrapper."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def checkpoint_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a transformer block through torch.utils.checkpoint.

    Checkpointing trades extra forward compute during backward for lower
    activation memory. The use_reentrant argument is preferred in modern
    PyTorch, with a fallback for older versions.
    """

    try:
        return checkpoint(block, x, use_reentrant=False)
    except TypeError:
        return checkpoint(block, x)

