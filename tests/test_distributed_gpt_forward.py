import os
import socket

import pytest
import torch

from nano_megatron_engine.model.config import GPTConfig
from nano_megatron_engine.model.distributed_gpt import DistributedGPTModel
from nano_megatron_engine.model.gpt import GPTModel
from nano_megatron_engine.parallel import init_distributed_from_env, is_distributed_available, is_distributed_initialized


def test_distributed_gpt_requires_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    config = _distributed_config()
    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        DistributedGPTModel(config)


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed GPT forward tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_gpt_forward_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_gpt_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_gpt_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_dense_gpt_logits_and_loss_match_distributed(world_size)
        _assert_tensor_parallel_size_mismatch_raises()
        _assert_invalid_vocab_divisibility_raises()
        _assert_invalid_sequence_length_raises(world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_dense_gpt_logits_and_loss_match_distributed(world_size: int) -> None:
    torch.manual_seed(1301)
    dense_config = _dense_config()
    distributed_config = _distributed_config(world_size=world_size)
    dense = GPTModel(dense_config)
    dense.eval()
    distributed = DistributedGPTModel(distributed_config)
    distributed.eval()
    distributed.copy_from_dense_(dense)

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

    assert distributed_logits.shape == dense_logits.shape == (2, 5, dense_config.vocab_size)
    assert dense_loss is not None
    assert distributed_loss is not None
    torch.testing.assert_close(distributed_logits, dense_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(distributed_loss, dense_loss, atol=1e-6, rtol=1e-5)


def _assert_tensor_parallel_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="distributed GPT.*tensor_parallel_size=1.*world_size=2"):
        DistributedGPTModel(_dense_config())


def _assert_invalid_vocab_divisibility_raises() -> None:
    config = GPTConfig(
        vocab_size=33,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    with pytest.raises(ValueError, match="distributed GPT.*strict vocab divisibility.*vocab_size=33.*world_size=2"):
        DistributedGPTModel(config)


def _assert_invalid_sequence_length_raises(world_size: int) -> None:
    distributed = DistributedGPTModel(_distributed_config(world_size=world_size))
    with pytest.raises(ValueError, match="distributed GPT sequence length 9 exceeds block_size 8"):
        distributed(torch.zeros(2, 9, dtype=torch.long))


def _dense_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=1,
    )


def _distributed_config(world_size: int = 2) -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.0,
        tensor_parallel_size=world_size,
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
