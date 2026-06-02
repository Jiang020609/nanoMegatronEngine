import math
import os
import socket

import pytest
import torch
from torch.nn import functional as F

from nano_megatron_engine.model import GPTConfig, GPTModel, build_distributed_gpt_from_dense
from nano_megatron_engine.parallel import RNGStateTracker, init_distributed_from_env, is_distributed_available


@pytest.mark.skipif(
    os.environ.get("NME_RUN_DISTRIBUTED_TESTS") != "1",
    reason="distributed GPT dropout equivalence tests are opt-in; set NME_RUN_DISTRIBUTED_TESTS=1",
)
def test_cpu_gloo_distributed_gpt_dropout_equivalence_smoke():
    if not is_distributed_available():
        pytest.skip("torch.distributed is not available")

    import torch.multiprocessing as mp

    world_size = 2
    port = _find_free_port()
    mp.spawn(_distributed_gpt_dropout_worker, args=(world_size, port), nprocs=world_size, join=True)


def _distributed_gpt_dropout_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("USE_LIBUV", "0")

    try:
        init_distributed_from_env("gloo")
        _assert_dropout_dense_reference_matches_distributed(rank, world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_dropout_dense_reference_matches_distributed(rank: int, world_size: int) -> None:
    torch.manual_seed(2001)
    attention_seed = 2002
    residual_seed = 2003
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=8,
        dropout=0.2,
    )
    dense = GPTModel(config)
    dense.train()
    tracker = _distributed_tracker(rank, attention_seed, residual_seed)
    distributed = build_distributed_gpt_from_dense(dense, rng_tracker=tracker)
    distributed.train()
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

    dense.zero_grad(set_to_none=True)
    distributed.zero_grad(set_to_none=True)
    reference_logits, reference_loss = _tracked_dropout_dense_reference(
        dense,
        input_ids,
        targets,
        world_size=world_size,
        attention_seed=attention_seed,
        residual_seed=residual_seed,
    )
    distributed_logits, distributed_loss = distributed(input_ids, targets)

    assert reference_loss is not None
    assert distributed_loss is not None
    torch.testing.assert_close(distributed_logits, reference_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(distributed_loss, reference_loss, atol=1e-6, rtol=1e-5)

    reference_loss.backward()
    distributed_loss.backward()
    distributed.synchronize_replicated_gradients_()
    _assert_local_sharded_gradients_match(dense, distributed)


def _tracked_dropout_dense_reference(
    model: GPTModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    world_size: int,
    attention_seed: int,
    residual_seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    config = model.config
    attention_trackers = [
        _tracker_with_state("attention_dropout", attention_seed + rank) for rank in range(world_size)
    ]
    residual_tracker = _tracker_with_state("residual_dropout", residual_seed)

    positions = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
    x = model.token_embedding(input_ids) + model.position_embedding(positions)
    x = _tracked_dropout(x, config.dropout, residual_tracker, "residual_dropout")

    for block in model.blocks:
        x = x + _tracked_attention(block.attn, block.ln_1(x), attention_trackers, residual_tracker, world_size)
        hidden = block.mlp.net[0](block.ln_2(x))
        hidden = block.mlp.net[1](hidden)
        mlp_output = block.mlp.net[2](hidden)
        x = x + _tracked_dropout(mlp_output, config.dropout, residual_tracker, "residual_dropout")

    x = model.ln_f(x)
    logits = model.lm_head(x)
    loss = None
    if targets is not None:
        logits_for_loss = logits[:, :-1, :].contiguous().view(-1, config.vocab_size)
        targets_for_loss = targets[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(logits_for_loss, targets_for_loss)
    return logits, loss


def _tracked_attention(
    attention: torch.nn.Module,
    x: torch.Tensor,
    attention_trackers: list[RNGStateTracker],
    residual_tracker: RNGStateTracker,
    world_size: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    hidden_size = attention.n_embd
    head_dim = attention.head_dim
    local_heads = attention.n_head // world_size
    local_hidden = local_heads * head_dim
    qkv = attention.qkv(x)
    query_all, key_all, value_all = qkv.split(hidden_size, dim=-1)
    local_contexts = []

    for rank, tracker in enumerate(attention_trackers):
        start = rank * local_hidden
        end = start + local_hidden
        query = _shape_heads(query_all[:, :, start:end], batch_size, seq_len, local_heads, head_dim)
        key = _shape_heads(key_all[:, :, start:end], batch_size, seq_len, local_heads, head_dim)
        value = _shape_heads(value_all[:, :, start:end], batch_size, seq_len, local_heads, head_dim)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(head_dim)
        causal_mask = attention.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = _tracked_dropout(weights, attention.dropout_p, tracker, "attention_dropout")
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, local_hidden)
        local_contexts.append(context)

    full_context = torch.cat(local_contexts, dim=-1)
    output = attention.proj(full_context)
    return _tracked_dropout(output, attention.dropout_p, residual_tracker, "residual_dropout")


def _assert_local_sharded_gradients_match(dense: GPTModel, distributed: torch.nn.Module) -> None:
    rank = distributed.rank
    start = distributed.token_embedding.local_vocab_start
    end = distributed.token_embedding.local_vocab_end
    _assert_grad_close(
        distributed.token_embedding.weight_shards[0],
        dense.token_embedding.weight.grad[start:end],
        "token_embedding.weight",
    )

    for dense_block, distributed_block in zip(dense.blocks, distributed.blocks):
        qkv = distributed_block.attn.qkv
        _assert_grad_close(qkv.weight, _local_qkv_slice(dense_block.attn.qkv.weight.grad, qkv), "attn.qkv.weight")
        _assert_grad_close(
            distributed_block.attn.proj.weight,
            dense_block.attn.proj.weight.grad[:, qkv.local_start : qkv.local_end],
            "attn.proj.weight",
        )
        fc1 = distributed_block.fc1
        dense_fc1 = dense_block.mlp.net[0]
        _assert_grad_close(
            fc1.weight,
            dense_fc1.weight.grad[fc1.local_out_start : fc1.local_out_end],
            "mlp.fc1.weight",
        )
        fc2 = distributed_block.fc2
        dense_fc2 = dense_block.mlp.net[2]
        _assert_grad_close(
            fc2.weight,
            dense_fc2.weight.grad[:, fc2.local_in_start : fc2.local_in_end],
            "mlp.fc2.weight",
        )

    assert rank in range(distributed.world_size)


def _local_qkv_slice(dense_tensor: torch.Tensor, qkv: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            dense_tensor[qkv.local_start : qkv.local_end],
            dense_tensor[qkv.hidden_size + qkv.local_start : qkv.hidden_size + qkv.local_end],
            dense_tensor[2 * qkv.hidden_size + qkv.local_start : 2 * qkv.hidden_size + qkv.local_end],
        ],
        dim=0,
    )


def _assert_grad_close(parameter: torch.nn.Parameter, expected: torch.Tensor, name: str) -> None:
    assert parameter.grad is not None, f"missing distributed gradient for {name}"
    assert expected is not None, f"missing reference gradient for {name}"
    torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)


def _shape_heads(
    x: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _tracked_dropout(
    x: torch.Tensor,
    dropout: float,
    tracker: RNGStateTracker,
    name: str,
) -> torch.Tensor:
    if dropout == 0.0:
        return x
    with tracker.fork(name):
        return F.dropout(x, p=dropout, training=True)


def _distributed_tracker(rank: int, attention_seed: int, residual_seed: int) -> RNGStateTracker:
    tracker = RNGStateTracker(device="cpu")
    tracker.add("attention_dropout", attention_seed + rank)
    tracker.add("residual_dropout", residual_seed)
    return tracker


def _tracker_with_state(name: str, seed: int) -> RNGStateTracker:
    tracker = RNGStateTracker(device="cpu")
    tracker.add(name, seed)
    return tracker


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
