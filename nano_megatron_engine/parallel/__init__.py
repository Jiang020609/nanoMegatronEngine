"""Educational tensor-parallel layers and collective helpers."""

from nano_megatron_engine.parallel.column_parallel_linear import ColumnParallelLinear
from nano_megatron_engine.parallel.collective_adapters import (
    DistributedRankLocalCollectives,
    FakeShardListCollectives,
    RankLocalCollectiveProtocol,
    ShardListCollectiveProtocol,
)
from nano_megatron_engine.parallel.distributed_collectives import (
    distributed_all_gather,
    distributed_all_reduce_sum,
    distributed_reduce_scatter_sum,
    get_backend,
    get_expected_device_type,
    get_rank,
    get_world_size,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
    validate_rank_local_tensor_device,
)
from nano_megatron_engine.parallel.distributed_column_parallel_linear import DistributedColumnParallelLinear
from nano_megatron_engine.parallel.distributed_qkv_parallel_linear import DistributedQKVParallelLinear
from nano_megatron_engine.parallel.distributed_row_parallel_linear import DistributedRowParallelLinear
from nano_megatron_engine.parallel.fake_tp import (
    fake_all_gather,
    fake_all_reduce_sum,
    fake_reduce_scatter_sum,
    partition_range,
    split_tensor_along_dim,
    validate_divisible,
)
from nano_megatron_engine.parallel.row_parallel_linear import RowParallelLinear
from nano_megatron_engine.parallel.rng import RNGState, RNGStateTracker, capture_rng_state, restore_rng_state
from nano_megatron_engine.parallel.vocab_parallel_embedding import VocabParallelEmbedding
from nano_megatron_engine.parallel.vocab_parallel_lm_head import VocabParallelLMHead

__all__ = [
    "ColumnParallelLinear",
    "DistributedColumnParallelLinear",
    "DistributedQKVParallelLinear",
    "DistributedRankLocalCollectives",
    "DistributedRowParallelLinear",
    "FakeShardListCollectives",
    "RankLocalCollectiveProtocol",
    "RowParallelLinear",
    "RNGState",
    "RNGStateTracker",
    "ShardListCollectiveProtocol",
    "VocabParallelEmbedding",
    "VocabParallelLMHead",
    "distributed_all_gather",
    "distributed_all_reduce_sum",
    "distributed_reduce_scatter_sum",
    "fake_all_gather",
    "fake_all_reduce_sum",
    "fake_reduce_scatter_sum",
    "capture_rng_state",
    "get_backend",
    "get_expected_device_type",
    "get_rank",
    "get_world_size",
    "init_distributed_from_env",
    "is_distributed_available",
    "is_distributed_initialized",
    "partition_range",
    "restore_rng_state",
    "split_tensor_along_dim",
    "validate_rank_local_tensor_device",
    "validate_divisible",
]
