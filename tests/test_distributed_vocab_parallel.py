import os
import socket

import pytest
import torch
from torch import nn

from nano_megatron_engine.parallel import (
    DistributedRankLocalCollectives,
    VocabParallelEmbedding,
    VocabParallelLMHead,
    init_distributed_from_env,
    is_distributed_available,
    is_distributed_initialized,
)


def test_distributed_vocab_modules_require_initialized_distributed():
    if is_distributed_initialized():
        pytest.skip("process group already initialized by the test environment")

    collectives = DistributedRankLocalCollectives()
    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        VocabParallelEmbedding(64, 8, tp_size=2, collectives=collectives)
    with pytest.raises(RuntimeError, match="torch.distributed.*init_distributed_from_env"):
        VocabParallelLMHead(8, 64, tp_size=2, collectives=collectives)


def test_vocab_parallel_modules_validate_collective_adapter_shape():
    bad_collectives = object()

    with pytest.raises(TypeError, match="DistributedRankLocalCollectives.*all_reduce_sum"):
        VocabParallelEmbedding(64, 8, tp_size=2, collectives=bad_collectives)
    with pytest.raises(TypeError, match="DistributedRankLocalCollectives.*all_gather"):
        VocabParallelLMHead(8, 64, tp_size=2, collectives=bad_collectives)


def test_vocab_parallel_dense_factory_helpers_validate_module_type():
    with pytest.raises(TypeError, match="nn.Embedding"):
        VocabParallelEmbedding.from_embedding(nn.Linear(8, 64))
    with pytest.raises(TypeError, match="nn.Linear"):
        VocabParallelLMHead.from_linear(nn.Embedding(64, 8))


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed vocab module tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_vocab_modules_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_vocab_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_vocab_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_extra_repr_and_local_vocab_ranges(world_size)
        _assert_embedding_output_matches_dense(world_size)
        _assert_embedding_gradients_match_dense_slice(world_size)
        _assert_lm_head_output_matches_dense(world_size)
        _assert_lm_head_gradients_match_dense(world_size)
        _assert_merge_helpers_gather_rank_local_weights(world_size)
        _assert_uneven_distributed_vocab_partitions_raise(world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_extra_repr_and_local_vocab_ranges(world_size: int) -> None:
    collectives = DistributedRankLocalCollectives()
    rank = collectives.get_rank()
    local_size = 64 // world_size
    expected_start = rank * local_size
    expected_end = expected_start + local_size

    embedding = VocabParallelEmbedding(
        64,
        8,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    lm_head = VocabParallelLMHead(
        8,
        64,
        tp_size=world_size,
        bias=True,
        collectives=DistributedRankLocalCollectives(),
    )

    for module in (embedding, lm_head):
        assert module.vocab_start == expected_start
        assert module.vocab_end == expected_end
        assert module.local_vocab_size == local_size
        assert module.local_vocab_start == expected_start
        assert module.local_vocab_end == expected_end
        text = module.extra_repr()
        assert "mode=distributed_rank_local" in text
        assert f"local_vocab_range=[{expected_start}, {expected_end})" in text
        assert f"local_vocab_size={local_size}" in text

    assert "vocab_size=64" in embedding.extra_repr()
    assert "embedding_dim=8" in embedding.extra_repr()
    assert "hidden_size=8" in lm_head.extra_repr()
    assert "bias=True" in lm_head.extra_repr()


def _assert_embedding_output_matches_dense(world_size: int) -> None:
    torch.manual_seed(910)
    dense = nn.Embedding(64, 8)
    layer = VocabParallelEmbedding.from_embedding(
        dense,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    input_ids = torch.tensor([[0, 1, 31, 32], [33, 44, 62, 63]])

    output = layer(input_ids)

    assert output.shape == (2, 4, 8)
    torch.testing.assert_close(output, dense(input_ids), atol=1e-6, rtol=1e-6)
    start, end = layer.local_vocab_start, layer.local_vocab_end
    torch.testing.assert_close(layer.weight_shards[0], dense.weight[start:end])


def _assert_embedding_gradients_match_dense_slice(world_size: int) -> None:
    torch.manual_seed(911)
    dense = nn.Embedding(64, 8)
    layer = VocabParallelEmbedding.from_embedding(
        dense,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    input_ids = torch.tensor([[0, 3, 31, 32], [33, 45, 62, 63]])

    dense_loss = dense(input_ids).square().mean()
    distributed_loss = layer(input_ids).square().mean()
    dense_loss.backward()
    distributed_loss.backward()

    start, end = layer.local_vocab_start, layer.local_vocab_end
    torch.testing.assert_close(layer.weight_shards[0].grad, dense.weight.grad[start:end], atol=1e-6, rtol=1e-6)


def _assert_lm_head_output_matches_dense(world_size: int) -> None:
    torch.manual_seed(912)
    dense = nn.Linear(8, 64, bias=True)
    layer = VocabParallelLMHead.from_linear(
        dense,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    x = torch.randn(2, 3, 8)

    logits = layer(x)

    assert logits.shape == (2, 3, 64)
    torch.testing.assert_close(logits, dense(x), atol=1e-6, rtol=1e-6)
    start, end = layer.local_vocab_start, layer.local_vocab_end
    torch.testing.assert_close(layer.weight_shards[0], dense.weight[start:end])
    assert layer.bias_shards is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias_shards[0], dense.bias[start:end])


def _assert_lm_head_gradients_match_dense(world_size: int) -> None:
    torch.manual_seed(913)
    dense = nn.Linear(8, 64, bias=True)
    layer = VocabParallelLMHead.from_linear(
        dense,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    dense_x = torch.randn(2, 3, 8, requires_grad=True)
    distributed_x = dense_x.detach().clone().requires_grad_()

    dense_loss = dense(dense_x).square().mean()
    distributed_loss = layer(distributed_x).square().mean()
    dense_loss.backward()
    distributed_loss.backward()

    start, end = layer.local_vocab_start, layer.local_vocab_end
    torch.testing.assert_close(distributed_x.grad, dense_x.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(layer.weight_shards[0].grad, dense.weight.grad[start:end], atol=1e-6, rtol=1e-6)
    assert layer.bias_shards is not None
    assert dense.bias is not None
    torch.testing.assert_close(layer.bias_shards[0].grad, dense.bias.grad[start:end], atol=1e-6, rtol=1e-6)


def _assert_merge_helpers_gather_rank_local_weights(world_size: int) -> None:
    torch.manual_seed(914)
    dense_embedding = nn.Embedding(64, 8)
    dense_lm_head = nn.Linear(8, 64, bias=True)
    embedding = VocabParallelEmbedding.from_embedding(
        dense_embedding,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )
    lm_head = VocabParallelLMHead.from_linear(
        dense_lm_head,
        tp_size=world_size,
        collectives=DistributedRankLocalCollectives(),
    )

    merged_embedding = embedding.merge_to_embedding()
    merged_lm_head = lm_head.merge_to_linear()

    torch.testing.assert_close(merged_embedding.weight, dense_embedding.weight)
    torch.testing.assert_close(merged_lm_head.weight, dense_lm_head.weight)
    assert merged_lm_head.bias is not None
    assert dense_lm_head.bias is not None
    torch.testing.assert_close(merged_lm_head.bias, dense_lm_head.bias)


def _assert_uneven_distributed_vocab_partitions_raise(world_size: int) -> None:
    collectives = DistributedRankLocalCollectives()
    with pytest.raises(ValueError, match="distributed vocab-parallel.*strict divisibility.*vocab_size=65.*world_size=2"):
        VocabParallelEmbedding(65, 8, tp_size=world_size, collectives=collectives)
    with pytest.raises(ValueError, match="distributed vocab-parallel.*strict divisibility.*vocab_size=65.*world_size=2"):
        VocabParallelLMHead(8, 65, tp_size=world_size, collectives=collectives)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
