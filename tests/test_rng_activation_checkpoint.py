from copy import deepcopy

import torch
from torch import nn

from nano_megatron_engine.memory.activation_checkpoint import checkpoint_block
from nano_megatron_engine.parallel import RNGStateTracker


class _DropoutBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(16, 8),
            nn.Dropout(0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_activation_checkpoint_preserves_dropout_replay_with_rng_tracker():
    torch.manual_seed(1601)
    direct = _DropoutBlock()
    checkpointed = deepcopy(direct)
    x = torch.randn(4, 8)
    direct_x = x.clone().requires_grad_()
    checkpoint_x = x.clone().requires_grad_()

    tracker = RNGStateTracker(device="cpu")
    tracker.add("dropout", 1602)
    dropout_state = tracker.get_state("dropout")

    with tracker.fork("dropout"):
        direct_y = direct(direct_x)

    tracker.set_state("dropout", dropout_state)
    with tracker.fork("dropout"):
        checkpoint_y = checkpoint_block(checkpointed, checkpoint_x)

    torch.testing.assert_close(checkpoint_y, direct_y)

    direct_loss = direct_y.square().mean()
    checkpoint_loss = checkpoint_y.square().mean()
    direct_loss.backward()
    checkpoint_loss.backward()

    torch.testing.assert_close(checkpoint_x.grad, direct_x.grad)
    for direct_parameter, checkpoint_parameter in zip(direct.parameters(), checkpointed.parameters()):
        assert direct_parameter.grad is not None
        assert checkpoint_parameter.grad is not None
        torch.testing.assert_close(checkpoint_parameter.grad, direct_parameter.grad)
