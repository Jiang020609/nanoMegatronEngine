import pytest
import torch

from nano_megatron_engine.parallel.fake_tp import (
    concat_tensor_parallel_outputs,
    partition_range,
    split_tensor_along_dim,
    sum_tensor_parallel_outputs,
    validate_divisible,
)


def test_split_and_concat_tensor_parallel_outputs():
    tensor = torch.arange(24).view(2, 3, 4)

    chunks = split_tensor_along_dim(tensor, num_chunks=2, dim=-1)
    merged = concat_tensor_parallel_outputs(chunks, dim=-1)

    assert len(chunks) == 2
    assert chunks[0].shape == (2, 3, 2)
    assert torch.equal(merged, tensor)


def test_sum_tensor_parallel_outputs():
    chunks = [torch.ones(2, 3), torch.full((2, 3), 2.0), torch.full((2, 3), 3.0)]

    summed = sum_tensor_parallel_outputs(chunks)

    assert torch.equal(summed, torch.full((2, 3), 6.0))


def test_validate_divisible_raises_clear_error():
    with pytest.raises(ValueError, match="hidden_size=10 must be divisible"):
        validate_divisible(10, 3, "hidden_size")

    with pytest.raises(ValueError, match="divisor for hidden_size must be positive"):
        validate_divisible(10, 0, "hidden_size")


def test_partition_range_covers_even_and_uneven_ranges():
    assert [partition_range(64, 2, idx) for idx in range(2)] == [(0, 32), (32, 64)]
    assert [partition_range(65, 2, idx) for idx in range(2)] == [(0, 33), (33, 65)]
    assert [partition_range(5, 3, idx) for idx in range(3)] == [(0, 2), (2, 4), (4, 5)]


def test_partition_range_invalid_inputs_raise():
    with pytest.raises(ValueError, match="num_partitions must be positive"):
        partition_range(10, 0, 0)

    with pytest.raises(ValueError, match="partition_idx=3 must be in"):
        partition_range(10, 3, 3)
