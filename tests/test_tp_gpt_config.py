import pytest

from nano_megatron_engine.model import GPTConfig


def test_tensor_parallel_size_defaults_to_one():
    config = GPTConfig()

    assert config.tensor_parallel_size == 1
    assert config.bias is True


def test_invalid_bias_type_raises():
    with pytest.raises(TypeError, match="bias must be bool"):
        GPTConfig(bias=1)


def test_invalid_tensor_parallel_size_raises():
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        GPTConfig(tensor_parallel_size=0)


def test_incompatible_mlp_intermediate_size_raises():
    with pytest.raises(ValueError, match="MLP intermediate_size=40 must be divisible"):
        GPTConfig(n_embd=10, n_head=2, tensor_parallel_size=3)


def test_valid_attention_head_tensor_parallel_config():
    config = GPTConfig(n_embd=16, n_head=4, tensor_parallel_size=2)

    assert config.tensor_parallel_size == 2


def test_incompatible_attention_heads_raise():
    with pytest.raises(ValueError, match="n_head=3 must be divisible by tensor_parallel_size=2"):
        GPTConfig(n_embd=24, n_head=3, tensor_parallel_size=2)
