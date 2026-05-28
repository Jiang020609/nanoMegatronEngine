import pytest

from nano_megatron_engine.model import GPTConfig


def test_tensor_parallel_size_defaults_to_one():
    config = GPTConfig()

    assert config.tensor_parallel_size == 1


def test_invalid_tensor_parallel_size_raises():
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        GPTConfig(tensor_parallel_size=0)


def test_incompatible_mlp_intermediate_size_raises():
    with pytest.raises(ValueError, match="MLP intermediate_size=40 must be divisible"):
        GPTConfig(n_embd=10, n_head=2, tensor_parallel_size=3)
