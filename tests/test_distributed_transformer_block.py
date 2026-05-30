import os
import socket

import pytest
import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_transformer_block import DistributedTransformerBlock
from nano_megatron_engine.model.transformer import TransformerBlock
from nano_megatron_engine.parallel import (
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_transformer_block_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        DistributedTransformerBlock(hidden_size=8, num_heads=4, block_size=8, mlp_hidden_size=16)


def test_distributed_transformer_block_validates_before_distributed():
    with pytest.raises(ValueError, match="distributed transformer block.*hidden_size=10.*num_heads=4"):
        DistributedTransformerBlock(hidden_size=10, num_heads=4, block_size=8, mlp_hidden_size=16)
    with pytest.raises(ValueError, match="distributed transformer block dropout"):
        DistributedTransformerBlock(hidden_size=8, num_heads=4, block_size=8, mlp_hidden_size=16, dropout=1.0)
    with pytest.raises(TypeError, match="distributed transformer block bias must be bool"):
        DistributedTransformerBlock(hidden_size=8, num_heads=4, block_size=8, mlp_hidden_size=16, bias=1)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed transformer block tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_transformer_block_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_block_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_block_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_dense_block_matches_distributed()
        _assert_invalid_num_heads_divisibility_raises()
        _assert_invalid_mlp_hidden_size_divisibility_raises()
        _assert_invalid_sequence_length_raises()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_dense_block_matches_distributed() -> None:
    torch.manual_seed(1201)
    hidden_size = 8
    num_heads = 4
    block_size = 8
    mlp_hidden_size = 4 * hidden_size
    config = GPTConfig(
        vocab_size=32,
        block_size=block_size,
        n_layer=1,
        n_head=num_heads,
        n_embd=hidden_size,
        dropout=0.0,
    )
    dense = TransformerBlock(config)
    dense.eval()
    layer = DistributedTransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=block_size,
        mlp_hidden_size=mlp_hidden_size,
        bias=True,
        dropout=0.0,
    )
    layer.eval()
    layer.copy_from_dense_(dense)
    x = torch.randn(2, 5, hidden_size)

    dense_y = dense(x)
    distributed_y = layer(x)

    assert distributed_y.shape == dense_y.shape == (2, 5, hidden_size)
    torch.testing.assert_close(distributed_y, dense_y, atol=1e-6, rtol=1e-5)


def _assert_invalid_num_heads_divisibility_raises() -> None:
    with pytest.raises(ValueError, match="distributed transformer block.*strict head divisibility.*num_heads=3"):
        DistributedTransformerBlock(hidden_size=6, num_heads=3, block_size=8, mlp_hidden_size=12)


def _assert_invalid_mlp_hidden_size_divisibility_raises() -> None:
    with pytest.raises(ValueError, match="distributed transformer block.*mlp_hidden_size=15.*world_size=2"):
        DistributedTransformerBlock(hidden_size=8, num_heads=4, block_size=8, mlp_hidden_size=15)


def _assert_invalid_sequence_length_raises() -> None:
    layer = DistributedTransformerBlock(hidden_size=8, num_heads=4, block_size=4, mlp_hidden_size=16)
    with pytest.raises(ValueError, match="distributed transformer block sequence length 5 exceeds block_size 4"):
        layer(torch.randn(2, 5, 8))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
