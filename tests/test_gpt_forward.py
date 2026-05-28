import torch

from nano_megatron_engine.memory import estimate_training_memory
from nano_megatron_engine.model import GPTConfig, GPTModel


def test_gpt_forward_shape_and_scalar_loss():
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (4, config.block_size))

    logits, loss = model(input_ids, targets=input_ids)

    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_memory_estimator_returns_positive_values():
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPTModel(config)

    estimate = estimate_training_memory(model)

    assert estimate.parameter_bytes > 0
    assert estimate.gradient_bytes > 0
    assert estimate.optimizer_state_bytes > 0
    assert estimate.total_bytes > estimate.parameter_bytes

