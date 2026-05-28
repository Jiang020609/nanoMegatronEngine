"""Train tiny GPT on a fixed random batch for a few CPU-friendly steps."""

from __future__ import annotations

import torch

from nano_megatron_engine.engine import Trainer
from nano_megatron_engine.model import GPTConfig, GPTModel
from nano_megatron_engine.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = Trainer(model, optimizer, micro_batch_size=4, grad_clip_norm=1.0, device=device)

    batch = torch.randint(0, config.vocab_size, (8, config.block_size), device=device)
    print(f"Training tiny GPT on {device} for 20 steps")
    first_loss = None
    final_loss = None
    for step in range(1, 21):
        result = trainer.train_step(batch)
        first_loss = result.loss if first_loss is None else first_loss
        final_loss = result.loss
        if step == 1 or step % 5 == 0:
            print(f"step={step:02d} loss={result.loss:.4f} tokens/sec={result.tokens_per_sec:.1f}")

    print(f"initial_loss={first_loss:.4f} final_loss={final_loss:.4f}")


if __name__ == "__main__":
    main()

