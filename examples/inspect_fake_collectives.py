"""Inspect single-process fake tensor-parallel collectives."""

from __future__ import annotations

import torch

from nano_megatron_engine.parallel.fake_tp import (
    fake_all_gather,
    fake_all_reduce_sum,
    fake_reduce_scatter_sum,
    partition_range,
)


def main() -> None:
    torch.manual_seed(2030)
    print("Fake tensor-parallel collective inspection")
    print("This is single-process fake TP: no torch.distributed, no multi-GPU communication, no speedup claims.")

    gather_shards = [torch.randn(2, 2), torch.randn(2, 3)]
    gathered = fake_all_gather(gather_shards, dim=-1)
    print(f"fake_all_gather shard_shapes={[tuple(shard.shape) for shard in gather_shards]}")
    print(f"fake_all_gather output_shape={tuple(gathered.shape)}")

    reduce_shards = [torch.ones(2, 4), torch.full((2, 4), 2.0), torch.full((2, 4), 3.0)]
    reduced = fake_all_reduce_sum(reduce_shards)
    print(f"fake_all_reduce_sum shard_shapes={[tuple(shard.shape) for shard in reduce_shards]}")
    print(f"fake_all_reduce_sum output={reduced.tolist()}")

    scatter_source = [torch.arange(10, dtype=torch.float32).view(2, 5), torch.ones(2, 5)]
    scattered = fake_reduce_scatter_sum(scatter_source, num_partitions=2, partition_idx=0, dim=-1)
    print(f"fake_reduce_scatter_sum output_shape={tuple(scattered.shape)}")
    print(f"fake_reduce_scatter_sum output={scattered.tolist()}")

    ranges = [partition_range(65, 2, idx) for idx in range(2)]
    print(f"partition_range total_size=65 num_partitions=2 ranges={ranges}")


if __name__ == "__main__":
    main()
