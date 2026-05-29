"""Explicit adapter boundaries for fake and distributed collectives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from nano_megatron_engine.parallel.distributed_collectives import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_rank,
    get_world_size,
    is_distributed_initialized,
)
from nano_megatron_engine.parallel.fake_tp import (
    fake_all_gather,
    fake_all_reduce_sum,
    fake_reduce_scatter_sum,
    partition_range,
)


class ShardListCollectiveProtocol(Protocol):
    """Protocol for single-process collectives that receive all fake shards."""

    def all_reduce_sum(self, shards: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sum a list of fake shard tensors."""

    def all_gather(self, shards: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
        """Concatenate a list of fake shard tensors."""

    def reduce_scatter_sum(
        self,
        shards: Sequence[torch.Tensor],
        num_partitions: int,
        partition_idx: int,
        dim: int = -1,
    ) -> torch.Tensor:
        """Sum fake shards and return one fake partition."""

    def partition_range(self, total_size: int, num_partitions: int, partition_idx: int) -> tuple[int, int]:
        """Return the contiguous range for one fake partition."""


class RankLocalCollectiveProtocol(Protocol):
    """Protocol for distributed collectives that receive one local rank tensor."""

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum this rank's tensor across distributed ranks."""

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Gather this rank's tensor across distributed ranks."""

    def reduce_scatter_sum(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sum rank-local tensors and return this rank's partition."""

    def get_rank(self) -> int:
        """Return the initialized distributed rank."""

    def get_world_size(self) -> int:
        """Return the initialized distributed world size."""

    def is_initialized(self) -> bool:
        """Return whether the distributed process group is initialized."""


@dataclass(frozen=True)
class FakeShardListCollectives:
    """Adapter for single-process fake TP collectives.

    One Python process owns every fake shard tensor and passes them as lists.
    This adapter is used to make the educational fake TP boundary explicit. It
    is not a real distributed backend and does not use torch.distributed.
    """

    def all_reduce_sum(self, shards: Sequence[torch.Tensor]) -> torch.Tensor:
        return fake_all_reduce_sum(shards)

    def all_gather(self, shards: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
        return fake_all_gather(shards, dim=dim)

    def reduce_scatter_sum(
        self,
        shards: Sequence[torch.Tensor],
        num_partitions: int,
        partition_idx: int,
        dim: int = -1,
    ) -> torch.Tensor:
        return fake_reduce_scatter_sum(
            shards,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
            dim=dim,
        )

    def partition_range(self, total_size: int, num_partitions: int, partition_idx: int) -> tuple[int, int]:
        return partition_range(total_size, num_partitions, partition_idx)


@dataclass(frozen=True)
class DistributedRankLocalCollectives:
    """Adapter for optional rank-local CPU/Gloo distributed collectives.

    Each process owns only its local tensor and torch.distributed performs the
    collective across ranks. This is a low-level CPU/Gloo wrapper boundary and
    is not yet integrated into GPT/model tensor parallelism.
    """

    group: object | None = None

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return distributed_all_reduce_sum(tensor, group=self.group)

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return distributed_all_gather(tensor, dim=dim, group=self.group)

    def reduce_scatter_sum(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return distributed_reduce_scatter_sum(tensor, dim=dim, group=self.group)

    def get_rank(self) -> int:
        return get_rank(group=self.group)

    def get_world_size(self) -> int:
        return get_world_size(group=self.group)

    def is_initialized(self) -> bool:
        return is_distributed_initialized()
