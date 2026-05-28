"""Helpers for splitting a batch into microbatches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def split_batch(batch: Any, micro_batch_size: int) -> list[Any]:
    """Split tensors or nested tensor containers along batch dimension 0."""

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    batch_size = get_batch_size(batch)
    microbatches = []
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        microbatches.append(_slice_batch(batch, start, end))
    return microbatches


def get_batch_size(batch: Any) -> int:
    """Find the leading tensor dimension in a batch."""

    if torch.is_tensor(batch):
        if batch.ndim == 0:
            raise ValueError("batch tensors must have a batch dimension")
        return int(batch.shape[0])

    if isinstance(batch, Mapping):
        sizes = [get_batch_size(value) for value in batch.values() if _contains_tensor(value)]
        return _single_batch_size(sizes)

    if _is_sequence(batch):
        sizes = [get_batch_size(value) for value in batch if _contains_tensor(value)]
        return _single_batch_size(sizes)

    raise TypeError("batch must be a tensor or a nested container containing tensors")


def _slice_batch(batch: Any, start: int, end: int) -> Any:
    if torch.is_tensor(batch):
        return batch[start:end]

    if isinstance(batch, Mapping):
        return type(batch)((key, _slice_batch(value, start, end) if _contains_tensor(value) else value) for key, value in batch.items())

    if _is_sequence(batch):
        sliced = [_slice_batch(value, start, end) if _contains_tensor(value) else value for value in batch]
        return type(batch)(sliced) if not isinstance(batch, tuple) else tuple(sliced)

    return batch


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if _is_sequence(value):
        return any(_contains_tensor(item) for item in value)
    return False


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _single_batch_size(sizes: list[int]) -> int:
    if not sizes:
        raise TypeError("batch container does not contain tensors")
    first = sizes[0]
    if any(size != first for size in sizes):
        raise ValueError("all tensors in a batch must share the same leading dimension")
    return first

