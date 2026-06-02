"""Dropout helpers that can use named RNG streams."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.rng import RNGStateTracker


class TrackedDropout(nn.Module):
    """Dropout with optional process-local named RNG tracking."""

    def __init__(
        self,
        p: float = 0.5,
        *,
        rng_tracker: RNGStateTracker | None = None,
        rng_name: str = "dropout",
    ) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"tracked dropout probability must be in [0.0, 1.0), got {p}")
        if rng_tracker is not None and not isinstance(rng_tracker, RNGStateTracker):
            raise TypeError(f"rng_tracker must be an RNGStateTracker or None, got {type(rng_tracker).__name__}")
        if not isinstance(rng_name, str):
            raise TypeError(f"tracked dropout rng_name must be a str, got {type(rng_name).__name__}")
        if not rng_name:
            raise ValueError("tracked dropout rng_name must be non-empty")

        self.p = p
        self.rng_tracker = rng_tracker
        self.rng_name = rng_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rng_tracker is None or not self.training or self.p == 0.0:
            return F.dropout(x, p=self.p, training=self.training)

        with self.rng_tracker.fork(self.rng_name):
            return F.dropout(x, p=self.p, training=True)

    def extra_repr(self) -> str:
        tracked = self.rng_tracker is not None
        return f"p={self.p}, tracked={tracked}, rng_name={self.rng_name!r}"


def make_rank_local_rng_tracker(
    name: str,
    seed: int,
    rank: int,
    *,
    device: torch.device | str | None = None,
) -> RNGStateTracker:
    """Create an RNG tracker stream seeded with ``seed + rank``."""

    if not isinstance(seed, int):
        raise TypeError(f"rank-local RNG seed must be an int, got {type(seed).__name__}")
    if not isinstance(rank, int):
        raise TypeError(f"rank-local RNG rank must be an int, got {type(rank).__name__}")
    if rank < 0:
        raise ValueError(f"rank-local RNG rank must be non-negative, got {rank}")

    tracker = RNGStateTracker(device=device)
    tracker.add(name, seed + rank)
    return tracker
