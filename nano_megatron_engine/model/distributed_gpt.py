"""Rank-local distributed GPT forward prototype."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.memory.activation_checkpoint import checkpoint_block
from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_transformer_block import DistributedTransformerBlock
from nano_megatron_engine.parallel import (
    DistributedColumnParallelLinear,
    DistributedQKVParallelLinear,
    DistributedRowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLMHead,
)
from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


class DistributedGPTModel(nn.Module):
    """Rank-local GPT forward prototype.

    This model composes the module-level distributed prototypes into a full
    forward path. It is not wired into ``GPTModel`` and is not a training
    engine. Each rank owns vocab shards and tensor-parallel layer shards, while
    replicated components such as position embeddings and LayerNorm are copied
    in full on every rank.
    """

    def __init__(
        self,
        config: GPTConfig,
        collectives: DistributedRankLocalCollectives | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
        self.rank = self.collectives.get_rank()
        self.world_size = self.collectives.get_world_size()
        if self.world_size <= 0:
            raise ValueError(f"distributed GPT world_size must be positive, got {self.world_size}")
        if config.tensor_parallel_size != self.world_size:
            raise ValueError(
                "distributed GPT requires config.tensor_parallel_size to match distributed world_size, "
                f"got tensor_parallel_size={config.tensor_parallel_size}, world_size={self.world_size}"
            )
        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                "distributed GPT currently requires strict vocab divisibility, "
                f"got vocab_size={config.vocab_size}, world_size={self.world_size}"
            )

        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
            tp_size=self.world_size,
            collectives=self.collectives,
        )
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                DistributedTransformerBlock(
                    hidden_size=config.n_embd,
                    num_heads=config.n_head,
                    block_size=config.block_size,
                    mlp_hidden_size=config.mlp_hidden_size,
                    bias=True,
                    dropout=config.dropout,
                    collectives=self.collectives,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = VocabParallelLMHead(
            config.n_embd,
            config.vocab_size,
            tp_size=self.world_size,
            bias=False,
            collectives=self.collectives,
        )
        self.apply(self._init_weights)
        self.lm_head.tie_weight_shards(self.token_embedding.weight_shards)

    def copy_from_dense_(self, dense_model: nn.Module) -> "DistributedGPTModel":
        """Copy dense GPT weights into this rank's distributed GPT shards."""

        dense_token_embedding = getattr(dense_model, "token_embedding", None)
        dense_position_embedding = getattr(dense_model, "position_embedding", None)
        dense_blocks = getattr(dense_model, "blocks", None)
        dense_ln_f = getattr(dense_model, "ln_f", None)
        dense_lm_head = getattr(dense_model, "lm_head", None)
        if not isinstance(dense_token_embedding, nn.Embedding):
            raise TypeError("dense GPT model must expose an nn.Embedding named token_embedding")
        if not isinstance(dense_position_embedding, nn.Embedding):
            raise TypeError("dense GPT model must expose an nn.Embedding named position_embedding")
        if not isinstance(dense_blocks, nn.ModuleList):
            raise TypeError("dense GPT model must expose a ModuleList named blocks")
        if not isinstance(dense_ln_f, nn.LayerNorm):
            raise TypeError("dense GPT model must expose an nn.LayerNorm named ln_f")
        if not isinstance(dense_lm_head, nn.Linear):
            raise TypeError("dense GPT model must expose an nn.Linear named lm_head")
        if dense_lm_head.weight is not dense_token_embedding.weight:
            raise ValueError("distributed GPT prototype currently expects dense GPT tied token/lm_head weights")
        if len(dense_blocks) != len(self.blocks):
            raise ValueError(f"dense GPT has {len(dense_blocks)} blocks, expected {len(self.blocks)}")

        start, end = self.token_embedding.local_vocab_start, self.token_embedding.local_vocab_end
        with torch.no_grad():
            self.token_embedding.weight_shards[0].copy_(dense_token_embedding.weight[start:end])
            self.position_embedding.weight.copy_(dense_position_embedding.weight)
            self.ln_f.weight.copy_(dense_ln_f.weight)
            self.ln_f.bias.copy_(dense_ln_f.bias)

        for distributed_block, dense_block in zip(self.blocks, dense_blocks):
            distributed_block.copy_from_dense_(dense_block)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"distributed GPT expected input_ids to be a torch.Tensor, got {type(input_ids).__name__}")
        if input_ids.ndim != 2:
            raise ValueError("distributed GPT input_ids must have shape [batch, sequence]")
        self.collectives.validate_tensor_device(input_ids, "DistributedGPTModel")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"distributed GPT sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("distributed GPT targets must have the same shape as input_ids")
        if targets is not None:
            self.collectives.validate_tensor_device(targets, "DistributedGPTModel targets")
        if targets is not None and seq_len < 2:
            raise ValueError("distributed GPT sequence length must be at least 2 when computing next-token loss")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            if self.config.use_activation_checkpointing and self.training:
                x = checkpoint_block(block, x)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_for_loss = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            targets_for_loss = targets[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return logits, loss

    def num_parameters(self) -> int:
        """Return this rank's local trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def replicated_parameter_names(self) -> tuple[str, ...]:
        """Return parameter names that are replicated on every distributed rank."""

        return tuple(name for name, _ in self._iter_replicated_parameters())

    def synchronize_replicated_gradients_(self) -> tuple[str, ...]:
        """Average gradients for parameters replicated on every rank.

        Tensor-parallel shards remain rank-local and are intentionally not
        synchronized here. This helper is for explicit distributed prototype
        smoke tests before a local optimizer step; it is not a full distributed
        training engine.
        """

        synchronized = []
        for name, parameter in self._iter_replicated_parameters():
            if parameter.grad is None:
                continue
            reduced_grad = self.collectives.all_reduce_sum(parameter.grad)
            reduced_grad.div_(self.world_size)
            parameter.grad.copy_(reduced_grad)
            synchronized.append(name)
        return tuple(synchronized)

    def local_shard_summary(self) -> dict[str, object]:
        """Return serializable metadata for this rank's local GPT shards."""

        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_parameter_count": self.num_parameters(),
            "activation_checkpointing": self.config.use_activation_checkpointing,
            "replicated_parameter_names": self.replicated_parameter_names(),
            "token_embedding": {
                "vocab_range": (self.token_embedding.local_vocab_start, self.token_embedding.local_vocab_end),
                "weight_shape": tuple(self.token_embedding.weight_shards[0].shape),
            },
            "position_embedding": {
                "replicated": True,
                "weight_shape": tuple(self.position_embedding.weight.shape),
            },
            "blocks": [self._block_shard_summary(index, block) for index, block in enumerate(self.blocks)],
            "final_layernorm": {
                "replicated": True,
                "weight_shape": tuple(self.ln_f.weight.shape),
            },
            "lm_head": {
                "vocab_range": (self.lm_head.local_vocab_start, self.lm_head.local_vocab_end),
                "weight_shape": tuple(self.lm_head.weight_shards[0].shape),
                "tied_to_token_embedding": self.lm_head.weight_shards[0] is self.token_embedding.weight_shards[0],
            },
        }

    def _iter_replicated_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield "position_embedding.weight", self.position_embedding.weight
        for index, block in enumerate(self.blocks):
            yield f"blocks.{index}.ln_1.weight", block.ln_1.weight
            yield f"blocks.{index}.ln_1.bias", block.ln_1.bias
            yield f"blocks.{index}.ln_2.weight", block.ln_2.weight
            yield f"blocks.{index}.ln_2.bias", block.ln_2.bias
            if block.attn.proj.bias is not None:
                yield f"blocks.{index}.attn.proj.bias", block.attn.proj.bias
            if block.fc2.bias is not None:
                yield f"blocks.{index}.fc2.bias", block.fc2.bias
        yield "ln_f.weight", self.ln_f.weight
        yield "ln_f.bias", self.ln_f.bias

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, block_size={self.config.block_size}, "
            f"n_layer={self.config.n_layer}, n_head={self.config.n_head}, n_embd={self.config.n_embd}, "
            f"rank={self.rank}, world_size={self.world_size}"
        )

    @staticmethod
    def _block_shard_summary(index: int, block: DistributedTransformerBlock) -> dict[str, object]:
        return {
            "index": index,
            "layernorms_replicated": True,
            "attention": {
                "local_heads": block.attn.local_heads,
                "head_dim": block.attn.head_dim,
                "qkv_weight_shape": tuple(block.attn.qkv.weight.shape),
                "proj_weight_shape": tuple(block.attn.proj.weight.shape),
            },
            "mlp": {
                "fc1_weight_shape": tuple(block.fc1.weight.shape),
                "fc2_weight_shape": tuple(block.fc2.weight.shape),
            },
        }

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (DistributedColumnParallelLinear, DistributedQKVParallelLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, DistributedRowParallelLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VocabParallelEmbedding):
            for weight in module.weight_shards:
                nn.init.normal_(weight, mean=0.0, std=0.02)
        elif isinstance(module, VocabParallelLMHead):
            for weight in module.weight_shards:
                nn.init.normal_(weight, mean=0.0, std=0.02)
            if module.bias_shards is not None:
                for bias in module.bias_shards:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
