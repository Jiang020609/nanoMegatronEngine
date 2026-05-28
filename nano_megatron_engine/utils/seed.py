"""Randomness helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

