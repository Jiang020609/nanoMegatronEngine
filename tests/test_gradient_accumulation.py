import torch

from nano_megatron_engine.engine import Trainer
from nano_megatron_engine.model import GPTConfig, GPTModel
from nano_megatron_engine.utils.seed import set_seed


def test_microbatch_gradient_accumulation_matches_full_batch_step():
    set_seed(7)
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    full_model = GPTModel(config)
    micro_model = GPTModel(config)
    micro_model.load_state_dict(full_model.state_dict())

    batch = torch.randint(0, config.vocab_size, (6, config.block_size))
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.05)
    micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.05)

    full_optimizer.zero_grad(set_to_none=True)
    _, full_loss = full_model(batch, targets=batch)
    full_loss.backward()
    full_optimizer.step()

    trainer = Trainer(micro_model, micro_optimizer, micro_batch_size=2)
    result = trainer.train_step(batch)

    assert abs(result.loss - float(full_loss.detach().item())) < 1e-5
    for full_parameter, micro_parameter in zip(full_model.parameters(), micro_model.parameters()):
        assert torch.allclose(full_parameter, micro_parameter, atol=1e-5, rtol=1e-5)

