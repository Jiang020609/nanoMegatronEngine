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
    get_backend,
    get_expected_device_type,
    get_rank,
    get_world_size,
    is_distributed_initialized,
    validate_rank_local_tensor_device,
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

    def get_backend(self) -> str:
        """Return the initialized distributed backend name."""

    def get_expected_device_type(self) -> str:
        """Return the tensor device type expected by the backend."""

    def validate_tensor_device(self, tensor: torch.Tensor, op_name: str) -> None:
        """Validate that a rank-local tensor is on the backend's device type."""


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
    """Adapter for optional rank-local distributed collectives.

    Each process owns only its local tensor and torch.distributed performs the
    collective across ranks. Gloo expects CPU tensors and NCCL expects CUDA
    tensors. This is a low-level wrapper boundary and is not the main GPTModel
    path.
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

    def get_backend(self) -> str:
        return get_backend(group=self.group)

    def get_expected_device_type(self) -> str:
        return get_expected_device_type(group=self.group)

    def validate_tensor_device(self, tensor: torch.Tensor, op_name: str) -> None:
        validate_rank_local_tensor_device(tensor, op_name, group=self.group)
