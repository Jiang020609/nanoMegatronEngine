import math
import os
import socket

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nano_megatron_engine.model.attention import CausalSelfAttention
from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_attention import DistributedCausalSelfAttention
from nano_megatron_engine.parallel import (
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_causal_attention_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        DistributedCausalSelfAttention(hidden_size=8, num_heads=4, block_size=8)


def test_distributed_causal_attention_validates_before_distributed():
    with pytest.raises(ValueError, match="distributed causal attention.*hidden_size=10.*num_heads=4"):
        DistributedCausalSelfAttention(hidden_size=10, num_heads=4, block_size=8)
    with pytest.raises(ValueError, match="distributed causal attention dropout"):
        DistributedCausalSelfAttention(hidden_size=8, num_heads=4, block_size=8, dropout=1.0)
    with pytest.raises(TypeError, match="distributed causal attention bias must be bool"):
        DistributedCausalSelfAttention(hidden_size=8, num_heads=4, block_size=8, bias=1)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed attention tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_causal_attention_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_attention_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_attention_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_dense_attention_matches_distributed()
        _assert_bias_false_attention_matches_distributed()
        _assert_qkv_ordering_regression()
        _assert_invalid_num_heads_divisibility_raises()
        _assert_invalid_sequence_length_raises()
        _assert_output_projection_bias_is_applied_once()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_dense_attention_matches_distributed() -> None:
    torch.manual_seed(1101)
    hidden_size = 8
    num_heads = 4
    block_size = 8
    config = GPTConfig(
        vocab_size=32,
        block_size=block_size,
        n_layer=1,
        n_head=num_heads,
        n_embd=hidden_size,
        dropout=0.0,
    )
    dense = CausalSelfAttention(config)
    dense.eval()
    layer = DistributedCausalSelfAttention(hidden_size, num_heads, block_size, bias=True, dropout=0.0)
    layer.eval()
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 5, hidden_size)

    dense_y = dense(x)
    distributed_y = layer(x)

    assert distributed_y.shape == dense_y.shape == (2, 5, hidden_size)
    torch.testing.assert_close(distributed_y, dense_y, atol=1e-6, rtol=1e-5)


def _assert_bias_false_attention_matches_distributed() -> None:
    torch.manual_seed(1102)
    hidden_size = 8
    num_heads = 4
    block_size = 8
    dense = _DenseCausalAttention(hidden_size, num_heads, block_size, bias=False, dropout=0.0)
    dense.eval()
    layer = DistributedCausalSelfAttention(hidden_size, num_heads, block_size, bias=False, dropout=0.0)
    layer.eval()
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 5, hidden_size)

    torch.testing.assert_close(layer(x), dense(x), atol=1e-6, rtol=1e-5)


def _assert_qkv_ordering_regression() -> None:
    hidden_size = 8
    num_heads = 4
    block_size = 8
    dense = _DenseCausalAttention(hidden_size, num_heads, block_size, bias=True, dropout=0.0)
    with torch.no_grad():
        dense.qkv.weight.copy_(
            torch.arange(3 * hidden_size * hidden_size, dtype=torch.float32).view(3 * hidden_size, hidden_size)
            / 100.0
        )
        dense.qkv.bias.copy_(torch.arange(3 * hidden_size, dtype=torch.float32) / 10.0)
        dense.proj.weight.copy_(
            torch.arange(hidden_size * hidden_size, dtype=torch.float32).view(hidden_size, hidden_size) / 50.0
        )
        dense.proj.bias.copy_(torch.linspace(-0.4, 0.4, hidden_size))
    dense.eval()
    layer = DistributedCausalSelfAttention(hidden_size, num_heads, block_size, bias=True, dropout=0.0)
    layer.eval()
    layer.copy_from_dense_(dense)
    x = torch.linspace(-1.0, 1.0, 2 * 5 * hidden_size).view(2, 5, hidden_size)

    torch.testing.assert_close(layer(x), dense(x), atol=1e-4, rtol=1e-5)


def _assert_invalid_num_heads_divisibility_raises() -> None:
    with pytest.raises(ValueError, match="distributed causal attention.*strict head divisibility.*num_heads=3.*world_size=2"):
        DistributedCausalSelfAttention(hidden_size=6, num_heads=3, block_size=8)


def _assert_invalid_sequence_length_raises() -> None:
    layer = DistributedCausalSelfAttention(hidden_size=8, num_heads=4, block_size=4)
    with pytest.raises(ValueError, match="distributed causal attention sequence length 5 exceeds block_size 4"):
        layer(torch.randn(2, 5, 8))


def _assert_output_projection_bias_is_applied_once() -> None:
    hidden_size = 8
    num_heads = 4
    block_size = 8
    dense = _DenseCausalAttention(hidden_size, num_heads, block_size, bias=True, dropout=0.0)
    expected_bias = torch.linspace(-1.0, 1.0, hidden_size)
    with torch.no_grad():
        dense.qkv.weight.zero_()
        dense.qkv.bias.zero_()
        dense.proj.weight.zero_()
        dense.proj.bias.copy_(expected_bias)
    layer = DistributedCausalSelfAttention(hidden_size, num_heads, block_size, bias=True, dropout=0.0)
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 3, hidden_size)

    expected = expected_bias.view(1, 1, hidden_size).expand(2, 3, hidden_size)
    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-6)


class _DenseCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, block_size: int, bias: bool, dropout: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = self._shape_heads(query, batch_size, seq_len)
        key = self._shape_heads(key, batch_size, seq_len)
        value = self._shape_heads(value, batch_size, seq_len)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.resid_dropout(self.proj(context))

    def _shape_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
