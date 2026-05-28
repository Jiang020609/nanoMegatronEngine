"""Helpers for educational single-process tensor parallelism."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_divisible(value: int, divisor: int, name: str) -> None:
    """Raise a clear error if value cannot be evenly sharded."""

    if divisor <= 0:
        raise ValueError(f"divisor for {name} must be positive, got {divisor}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by divisor={divisor}")


def split_tensor_along_dim(tensor: torch.Tensor, num_chunks: int, dim: int) -> tuple[torch.Tensor, ...]:
    """Split a tensor into equal chunks along one dimension."""

    dim = _normalize_dim(dim, tensor.ndim)
    validate_divisible(tensor.shape[dim], num_chunks, f"tensor.shape[{dim}]")
    return tuple(chunk.contiguous() for chunk in torch.chunk(tensor, num_chunks, dim=dim))


def concat_tensor_parallel_outputs(chunks: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
    """Concatenate fake tensor-parallel outputs."""

    if not chunks:
        raise ValueError("chunks must not be empty")
    return torch.cat(tuple(chunks), dim=dim)


def sum_tensor_parallel_outputs(chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum fake tensor-parallel partial outputs."""

    if not chunks:
        raise ValueError("chunks must not be empty")
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("tensor must have at least one dimension")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for tensor with {ndim} dimensions")
    return dim

