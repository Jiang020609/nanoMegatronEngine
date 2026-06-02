"""Factory helpers for isolated distributed GPT prototypes."""

from __future__ import annotations

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import RNGStateTracker
from nano_megatron_engine.parallel.collective_adapters import DistributedRankLocalCollectives


def distributed_gpt_config_from_dense(config: GPTConfig, world_size: int) -> GPTConfig:
    """Return a distributed GPT config matching a dense config."""

    if not isinstance(config, GPTConfig):
        raise TypeError(f"config must be a GPTConfig, got {type(config).__name__}")
    if not isinstance(world_size, int):
        raise TypeError(f"world_size must be an int, got {type(world_size).__name__}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if config.tensor_parallel_size != 1:
        raise ValueError(
            "distributed GPT factory expects a dense config with tensor_parallel_size=1, "
            f"got {config.tensor_parallel_size}"
        )

    return GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
        use_activation_checkpointing=config.use_activation_checkpointing,
        tensor_parallel_size=world_size,
    )


def build_distributed_gpt_from_dense(
    dense_model: GPTModel,
    *,
    collectives: DistributedRankLocalCollectives | None = None,
    copy_weights: bool = True,
    rng_tracker: RNGStateTracker | None = None,
) -> DistributedGPTModel:
    """Build an isolated distributed GPT prototype from a dense GPT model."""

    if not isinstance(dense_model, GPTModel):
        raise TypeError(f"dense_model must be a GPTModel, got {type(dense_model).__name__}")
    if not isinstance(copy_weights, bool):
        raise TypeError(f"copy_weights must be bool, got {type(copy_weights).__name__}")
    if rng_tracker is not None and not isinstance(rng_tracker, RNGStateTracker):
        raise TypeError(f"rng_tracker must be an RNGStateTracker or None, got {type(rng_tracker).__name__}")

    distributed_collectives = collectives if collectives is not None else DistributedRankLocalCollectives()
    config = distributed_gpt_config_from_dense(
        dense_model.config,
        world_size=distributed_collectives.get_world_size(),
    )
    distributed = DistributedGPTModel(config, collectives=distributed_collectives, rng_tracker=rng_tracker)
    if copy_weights:
        distributed.copy_from_dense_(dense_model)
    return distributed
