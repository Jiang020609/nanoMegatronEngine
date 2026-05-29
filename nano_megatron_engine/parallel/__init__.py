"""Educational tensor-parallel layers and collective helpers."""

from nano_megatron_engine.parallel.column_parallel_linear import ColumnParallelLinear
from nano_megatron_engine.parallel.distributed_collectives import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_rank,
    get_world_size,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)
from nano_megatron_engine.parallel.fake_tp import (
    fake_all_gather,
    fake_all_reduce_sum,
    fake_reduce_scatter_sum,
    partition_range,
    split_tensor_along_dim,
    validate_divisible,
)
from nano_megatron_engine.parallel.row_parallel_linear import RowParallelLinear
from nano_megatron_engine.parallel.vocab_parallel_embedding import VocabParallelEmbedding
from nano_megatron_engine.parallel.vocab_parallel_lm_head import VocabParallelLMHead

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelLMHead",
    "distributed_all_gather",
    "distributed_all_reduce_sum",
    "distributed_reduce_scatter_sum",
    "fake_all_gather",
    "fake_all_reduce_sum",
    "fake_reduce_scatter_sum",
    "get_rank",
    "get_world_size",
    "init_distributed_from_env",
    "is_distributed_available",
    "is_distributed_initialized",
    "partition_range",
    "split_tensor_along_dim",
    "validate_divisible",
]
