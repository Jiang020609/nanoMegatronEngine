"""Tiny GPT language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.memory.activation_checkpoint import checkpoint_block
from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.transformer import TransformerBlock
from nano_megatron_engine.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLMHead,
)


class GPTModel(nn.Module):
    """A compact GPT-style next-token language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        if config.tensor_parallel_size == 1:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            self.token_embedding = VocabParallelEmbedding(
                config.vocab_size,
                config.n_embd,
                tp_size=config.tensor_parallel_size,
            )
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        if config.tensor_parallel_size == 1:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = VocabParallelLMHead(
                config.n_embd,
                config.vocab_size,
                tp_size=config.tensor_parallel_size,
                bias=False,
            )

        self.apply(self._init_weights)
        if config.tensor_parallel_size == 1:
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head.tie_weight_shards(self.token_embedding.weight_shards)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional next-token cross entropy loss."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if targets is not None and seq_len < 2:
            raise ValueError("sequence length must be at least 2 when computing next-token loss")

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
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ColumnParallelLinear):
            for weight in module.weight_shards:
                nn.init.normal_(weight, mean=0.0, std=0.02)
            if module.bias_shards is not None:
                for bias in module.bias_shards:
                    nn.init.zeros_(bias)
        elif isinstance(module, RowParallelLinear):
            for weight in module.weight_shards:
                nn.init.normal_(weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, CausalSelfAttention):
            if hasattr(module, "qkv_weight_shards"):
                for weight in module.qkv_weight_shards:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
                for bias in module.qkv_bias_shards:
                    nn.init.zeros_(bias)
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
