import torch

from nano_megatron_engine.model import GPTConfig, GPTModel


def test_gpt_forward_with_and_without_fake_tp_mlp():
    input_ids = torch.randint(0, 32, (2, 8))

    for tensor_parallel_size in (1, 2):
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            tensor_parallel_size=tensor_parallel_size,
        )
        model = GPTModel(config)

        logits, loss = model(input_ids, targets=input_ids)

        assert logits.shape == (2, 8, 32)
        assert loss is not None
        assert loss.ndim == 0


def test_gpt_fake_tp_mlp_backward_smoke():
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))

    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)
