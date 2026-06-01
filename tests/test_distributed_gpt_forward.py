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
        _assert_distributed_gpt_local_shard_summary(world_size)
        _assert_replicated_gradient_sync_averages_only_replicated_parameters(world_size)
        _assert_distributed_gpt_loss_backward_smoke(world_size)
        _assert_distributed_gpt_optimizer_step_smoke(world_size)
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


def _assert_distributed_gpt_local_shard_summary(world_size: int) -> None:
    distributed = DistributedGPTModel(_distributed_config(world_size=world_size))
    summary = distributed.local_shard_summary()
    rank = distributed.collectives.get_rank()
    local_vocab_size = distributed.config.vocab_size // world_size
    start = rank * local_vocab_size
    end = start + local_vocab_size

    assert summary["rank"] == rank
    assert summary["world_size"] == world_size
    assert summary["local_parameter_count"] == distributed.num_parameters()
    assert summary["replicated_parameter_names"] == distributed.replicated_parameter_names()
    assert "position_embedding.weight" in distributed.replicated_parameter_names()
    assert "blocks.0.attn.proj.bias" in distributed.replicated_parameter_names()
    assert "blocks.0.fc2.bias" in distributed.replicated_parameter_names()
    assert "token_embedding.weight_shards.0" not in distributed.replicated_parameter_names()
    assert summary["token_embedding"] == {
        "vocab_range": (start, end),
        "weight_shape": (local_vocab_size, distributed.config.n_embd),
    }
    assert summary["position_embedding"] == {
        "replicated": True,
        "weight_shape": (distributed.config.block_size, distributed.config.n_embd),
    }
    assert summary["final_layernorm"] == {
        "replicated": True,
        "weight_shape": (distributed.config.n_embd,),
    }
    assert summary["lm_head"] == {
        "vocab_range": (start, end),
        "weight_shape": (local_vocab_size, distributed.config.n_embd),
        "tied_to_token_embedding": True,
    }
    blocks = summary["blocks"]
    assert isinstance(blocks, list)
    assert len(blocks) == distributed.config.n_layer
    first_block = blocks[0]
    assert first_block["attention"]["local_heads"] == distributed.config.n_head // world_size
    assert first_block["attention"]["qkv_weight_shape"] == (3 * distributed.config.n_embd // world_size, distributed.config.n_embd)
    assert first_block["mlp"]["fc1_weight_shape"] == (
        distributed.config.mlp_hidden_size // world_size,
        distributed.config.n_embd,
    )
    assert first_block["mlp"]["fc2_weight_shape"] == (
        distributed.config.n_embd,
        distributed.config.mlp_hidden_size // world_size,
    )


def _assert_replicated_gradient_sync_averages_only_replicated_parameters(world_size: int) -> None:
    distributed = DistributedGPTModel(_distributed_config(world_size=world_size))
    rank = distributed.collectives.get_rank()
    rank_value = float(rank + 1)
    expected_replicated_value = float(sum(range(1, world_size + 1)) / world_size)
    replicated_names = set(distributed.replicated_parameter_names())

    for _, parameter in distributed.named_parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, rank_value)

    synchronized = distributed.synchronize_replicated_gradients_()
    assert set(synchronized) == replicated_names

    for name, parameter in distributed.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None
        expected_value = expected_replicated_value if name in replicated_names else rank_value
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter.grad, expected_value))


def _assert_distributed_gpt_loss_backward_smoke(world_size: int) -> None:
    torch.manual_seed(1302)
    dense = GPTModel(_dense_config())
    distributed = DistributedGPTModel(_distributed_config(world_size=world_size))
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

    _, dense_loss = dense(input_ids, targets)
    _, distributed_loss = distributed(input_ids, targets)
    assert dense_loss is not None
    assert distributed_loss is not None

    dense_loss.backward()
    distributed_loss.backward()

    _assert_all_trainable_grads_are_finite(distributed)
    start = distributed.token_embedding.local_vocab_start
    end = distributed.token_embedding.local_vocab_end
    assert dense.token_embedding.weight.grad is not None
    assert distributed.token_embedding.weight_shards[0].grad is not None
    assert distributed.token_embedding.weight_shards[0].grad.shape == dense.token_embedding.weight.grad[start:end].shape
    assert distributed.token_embedding.weight_shards[0].grad.abs().sum().item() > 0.0


def _assert_distributed_gpt_optimizer_step_smoke(world_size: int) -> None:
    torch.manual_seed(1303)
    distributed = DistributedGPTModel(_distributed_config(world_size=world_size))
    distributed.train()
    optimizer = torch.optim.SGD(distributed.parameters(), lr=1e-3)
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

    before = [parameter.detach().clone() for parameter in distributed.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    _, loss = distributed(input_ids, targets)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    _assert_all_trainable_grads_are_finite(distributed)
    synchronized = distributed.synchronize_replicated_gradients_()
    assert set(synchronized) == set(distributed.replicated_parameter_names())
    optimizer.step()

    after = [parameter.detach() for parameter in distributed.parameters() if parameter.requires_grad]
    assert len(before) == len(after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    assert all(torch.isfinite(parameter).all() for parameter in after)
    with torch.no_grad():
        _, post_step_loss = distributed(input_ids, targets)
    assert post_step_loss is not None
    assert torch.isfinite(post_step_loss)


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


def _assert_all_trainable_grads_are_finite(model: torch.nn.Module) -> None:
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
