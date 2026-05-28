import torch

from nano_megatron_engine.model import GPTConfig, GPTModel


def test_activation_checkpoint_forward_backward():
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        use_activation_checkpointing=True,
    )
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))

    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert loss.ndim == 0
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)

