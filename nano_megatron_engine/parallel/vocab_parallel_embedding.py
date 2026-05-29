"""Vocab-parallel token embedding modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.parallel.collective_adapters import (
    DistributedRankLocalCollectives,
    FakeShardListCollectives,
    ShardListCollectiveProtocol,
)
from nano_megatron_engine.parallel.fake_tp import partition_range


class VocabParallelEmbedding(nn.Module):
    """Split embedding rows across fake or rank-local distributed vocab shards."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 2,
        collectives: ShardListCollectiveProtocol | DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")

        self.collectives = collectives if collectives is not None else FakeShardListCollectives()
        self.is_rank_local = isinstance(self.collectives, DistributedRankLocalCollectives)
        self.rank = self.collectives.get_rank() if self.is_rank_local else 0
        self.world_size = self.collectives.get_world_size() if self.is_rank_local else tp_size
        if self.is_rank_local and tp_size != self.world_size:
            raise ValueError(
                f"distributed vocab embedding tp_size={tp_size} must match world_size={self.world_size}"
            )
        if self.is_rank_local and num_embeddings % self.world_size != 0:
            raise ValueError(
                "distributed vocab embedding currently requires divisible vocab partitions, "
                f"got num_embeddings={num_embeddings}, world_size={self.world_size}"
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = self.world_size if self.is_rank_local else tp_size
        self.vocab_ranges = [partition_range(num_embeddings, self.tp_size, idx) for idx in range(self.tp_size)]
        self.local_vocab_start, self.local_vocab_end = self.vocab_ranges[self.rank]
        parameter_ranges = [self.vocab_ranges[self.rank]] if self.is_rank_local else self.vocab_ranges
        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(end - start, embedding_dim)) for start, end in parameter_ranges]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight_shards:
            nn.init.normal_(weight, mean=0.0, std=0.02)

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        tp_size: int = 2,
        collectives: ShardListCollectiveProtocol | DistributedRankLocalCollectives | None = None,
    ) -> "VocabParallelEmbedding":
        layer = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            tp_size=tp_size,
            collectives=collectives,
        )
        layer.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
        with torch.no_grad():
            if layer.is_rank_local:
                layer.weight_shards[0].copy_(embedding.weight[layer.local_vocab_start : layer.local_vocab_end])
            else:
                for weight_shard, (start, end) in zip(layer.weight_shards, layer.vocab_ranges):
                    weight_shard.copy_(embedding.weight[start:end])
        return layer

    def merge_to_embedding(self) -> nn.Embedding:
        if self.is_rank_local:
            weight = self.collectives.all_gather(self.weight_shards[0], dim=0)
        else:
            weight = torch.cat(list(self.weight_shards), dim=0)
        embedding = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            embedding.weight.copy_(weight)
        return embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.is_rank_local:
            return self._forward_rank_local(input_ids)

        first_weight = self.weight_shards[0]
        local_outputs = []
        for weight, (start, end) in zip(self.weight_shards, self.vocab_ranges):
            local_output = torch.zeros(
                *input_ids.shape,
                self.embedding_dim,
                device=first_weight.device,
                dtype=first_weight.dtype,
            )
            if end == start:
                local_outputs.append(local_output)
                continue
            mask = (input_ids >= start) & (input_ids < end)
            local_ids = (input_ids - start).masked_fill(~mask, 0)
            local_output = F.embedding(local_ids, weight)
            local_outputs.append(local_output * mask.unsqueeze(-1).to(local_output.dtype))
        return self.collectives.all_reduce_sum(local_outputs)

    def _forward_rank_local(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.device.type != "cpu":
            raise ValueError(
                "VocabParallelEmbedding distributed path supports CPU/Gloo tensors only, "
                f"got {input_ids.device}"
            )

        weight = self.weight_shards[0]
        local_output = torch.zeros(
            *input_ids.shape,
            self.embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        if self.local_vocab_end == self.local_vocab_start:
            return self.collectives.all_reduce_sum(local_output)

        mask = (input_ids >= self.local_vocab_start) & (input_ids < self.local_vocab_end)
        local_ids = (input_ids - self.local_vocab_start).masked_fill(~mask, 0)
        local_output = F.embedding(local_ids, weight)
        local_output = local_output * mask.unsqueeze(-1).to(local_output.dtype)
        return self.collectives.all_reduce_sum(local_output)

    def extra_repr(self) -> str:
        mode = "distributed_rank_local" if self.is_rank_local else "fake_shard_list"
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"tp_size={self.tp_size}, mode={mode}"
        )
