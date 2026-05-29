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


def partition_range(total_size: int, num_partitions: int, partition_idx: int) -> tuple[int, int]:
    """Return a contiguous index range for one fake TP partition.

    The first partitions receive one extra element when total_size is not
    divisible, so all indices are covered exactly once.
    """

    if total_size < 0:
        raise ValueError(f"total_size must be non-negative, got {total_size}")
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}")
    if partition_idx < 0 or partition_idx >= num_partitions:
        raise ValueError(f"partition_idx={partition_idx} must be in [0, {num_partitions})")

    base = total_size // num_partitions
    remainder = total_size % num_partitions
    start = partition_idx * base + min(partition_idx, remainder)
    size = base + (1 if partition_idx < remainder else 0)
    return start, start + size


def split_tensor_along_dim(tensor: torch.Tensor, num_chunks: int, dim: int) -> tuple[torch.Tensor, ...]:
    """Split a tensor into equal chunks along one dimension."""

    dim = _normalize_dim(dim, tensor.ndim)
    validate_divisible(tensor.shape[dim], num_chunks, f"tensor.shape[{dim}]")
    return tuple(chunk.contiguous() for chunk in torch.chunk(tensor, num_chunks, dim=dim))


def fake_all_reduce_sum(shards: Sequence[torch.Tensor]) -> torch.Tensor:
    """Simulate a tensor-parallel all-reduce sum in one process."""

    shards = _as_tuple(shards)
    reference_shape = tuple(shards[0].shape)
    for shard in shards[1:]:
        if tuple(shard.shape) != reference_shape:
            raise ValueError(f"all-reduce shards must have matching shapes, got {reference_shape} and {tuple(shard.shape)}")

    total = shards[0]
    for shard in shards[1:]:
        total = total + shard
    return total


def fake_all_gather(shards: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Simulate gathering tensor-parallel shards by concatenating them."""

    shards = _as_tuple(shards)
    dim = _normalize_dim(dim, shards[0].ndim)
    reference_shape = tuple(shards[0].shape)
    for shard in shards[1:]:
        if shard.ndim != shards[0].ndim:
            raise ValueError("all-gather shards must have the same rank")
        for axis, (actual, expected) in enumerate(zip(shard.shape, reference_shape)):
            if axis != dim and actual != expected:
                raise ValueError(
                    "all-gather shards must match on non-gather dimensions, "
                    f"got dim {axis}: {actual} != {expected}"
                )
    return torch.cat(shards, dim=dim)


def fake_reduce_scatter_sum(
    shards: Sequence[torch.Tensor],
    num_partitions: int,
    partition_idx: int,
    dim: int = -1,
) -> torch.Tensor:
    """Simulate reduce-scatter by summing shards then returning one partition."""

    reduced = fake_all_reduce_sum(shards)
    dim = _normalize_dim(dim, reduced.ndim)
    start, end = partition_range(reduced.shape[dim], num_partitions, partition_idx)
    index = [slice(None)] * reduced.ndim
    index[dim] = slice(start, end)
    return reduced[tuple(index)]


def concat_tensor_parallel_outputs(chunks: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
    """Concatenate fake tensor-parallel outputs."""

    return fake_all_gather(chunks, dim=dim)


def sum_tensor_parallel_outputs(chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum fake tensor-parallel partial outputs."""

    return fake_all_reduce_sum(chunks)


def _as_tuple(shards: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    shards = tuple(shards)
    if not shards:
        raise ValueError("shards must not be empty")
    return shards


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("tensor must have at least one dimension")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for tensor with {ndim} dimensions")
    return dim
