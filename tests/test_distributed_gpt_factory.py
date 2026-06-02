import os
import socket

import pytest
import torch

from nano_megatron_engine.model import (
    GPTConfig,
    GPTModel,
    build_distributed_gpt_from_dense,
    distributed_gpt_config_from_dense,
)
from nano_megatron_engine.parallel import init_distributed_from_env, is_distributed_available, is_distributed_initialized


def test_distributed_gpt_config_from_dense_preserves_dense_fields():
    dense_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=16,
        bias=False,
        dropout=0.25,
        use_activation_checkpointing=True,
        tensor_parallel_size=1,
    )

    distributed_config = distributed_gpt_config_from_dense(dense_config, world_size=2)

    assert distributed_config.vocab_size == dense_config.vocab_size
    assert distributed_config.block_size == dense_config.block_size
    assert distributed_config.n_layer == dense_config.n_layer
    assert distributed_config.n_head == dense_config.n_head
    assert distributed_config.n_embd == dense_config.n_embd
    assert distributed_config.bias is False
    assert distributed_config.dropout == dense_config.dropout
    assert distributed_config.use_activation_checkpointing is True
    assert distributed_config.tensor_parallel_size == 2


def test_distributed_gpt_config_from_dense_rejects_non_dense_config():
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=4, n_embd=8, tensor_parallel_size=2)

    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        distributed_gpt_config_from_dense(config, world_size=2)


def test_build_distributed_gpt_from_dense_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    dense = GPTModel(GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=4, n_embd=8))
    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        build_distributed_gpt_from_dense(dense)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed GPT factory tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_gpt_factory_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_gpt_factory_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_gpt_factory_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_factory_builds_no_bias_model_from_dense(world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_factory_builds_no_bias_model_from_dense(world_size: int) -> None:
    torch.manual_seed(1901)
    dense_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=8,
        bias=False,
        dropout=0.0,
    )
    dense = GPTModel(dense_config)
    dense.eval()
    distributed = build_distributed_gpt_from_dense(dense)
    distributed.eval()

    input_ids = torch.tensor(
        [
            [0, 1, 16, 31, 4],
            [17, 2, 30, 8, 15],
        ]
    )
    targets = torch.tensor(
        [
            [1, 16, 31, 4, 5],
            [2, 30, 8, 15, 0],
        ]
    )

    dense_logits, dense_loss = dense(input_ids, targets)
    distributed_logits, distributed_loss = distributed(input_ids, targets)

    assert distributed.config.tensor_parallel_size == world_size
    assert distributed.config.bias is False
    assert distributed.blocks[0].attn.qkv.bias is None
    assert distributed.blocks[0].fc1.bias is None
    assert dense_loss is not None
    assert distributed_loss is not None
    torch.testing.assert_close(distributed_logits, dense_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(distributed_loss, dense_loss, atol=1e-6, rtol=1e-5)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
