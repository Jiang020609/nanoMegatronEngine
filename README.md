# nanoMegatronEngine

nanoMegatronEngine is a slim Megatron-style dense GPT training engine. It is
designed as a clean, readable training path for learning how a tiny
GPT-language-model stack fits together.

It is not a replacement for Megatron-LM and does not claim performance parity.
The v0.1 goal is a CPU-runnable, PyTorch-only core that is easy to inspect.

## What v0.1 Implements

- Tiny GPT model with token embeddings, learned positional embeddings, causal
  self-attention, MLP blocks, LayerNorm, residual connections, and an LM head.
- Next-token cross entropy loss.
- Microbatch splitting and gradient accumulation.
- A small single-device trainer with optional gradient clipping and tokens/sec
  reporting.
- Optional activation checkpointing for transformer blocks.
- Educational memory estimates for parameters, gradients, and optimizer state.
- A throughput/memory benchmark that runs on CPU and reports CUDA peak memory
  when CUDA is available.
- Pytest coverage for forward pass, scalar loss, microbatching, gradient
  accumulation, activation checkpointing, and memory estimates.

## What v0.1 Does Not Implement

- Tensor parallelism.
- Pipeline parallelism.
- ZeRO or optimizer sharding.
- MoE.
- FP8.
- Distributed training, NCCL, or multi-GPU execution.
- FlashAttention or custom CUDA kernels.

Those features are intentionally outside v0.1. This repository focuses on the
small dense-GPT path: one model, one device, readable code, and tests that make
the training mechanics explicit.

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Examples

```bash
python examples/train_tiny_gpt.py
python examples/train_tiny_gpt.py --tensor-parallel-size 2
python examples/bench_microbatch.py
python examples/bench_parallel_linear.py
python examples/compare_tp_mlp.py
python examples/compare_tp_attention.py
```

## v0.2 Fake Tensor Parallelism

Tensor parallelism splits one large layer across multiple ranks. In real
Megatron-style tensor parallelism, each rank owns only its shard of the
parameters and collective communication moves partial results between ranks.

v0.2 adds a correctness-first, single-process fake version for education:

- `ColumnParallelLinear` splits `nn.Linear.weight` along the output dimension
  (`dim=0`). Each shard produces local output features, and the default path
  concatenates those outputs along the last dimension.
- `RowParallelLinear` splits `nn.Linear.weight` along the input dimension
  (`dim=1`). The input is split the same way, each shard computes a partial
  output, and the partial outputs are summed. Bias is added once.
- `fake_tp.py` contains small helpers for divisible sharding, split, concat,
  and sum operations.

This is fake tensor parallelism because it runs all shards in one Python
process. It is meant to explain the math and test equivalence with
`torch.nn.Linear`, not to speed up training.

Real distributed tensor parallelism would add all-gather, all-reduce, process
groups, rank-local parameters, and integration with `torch.distributed`.

Run the full suite and the linear benchmark with:

```bash
pytest
python examples/bench_parallel_linear.py
```

## v0.3 Fake Tensor Parallel MLP

v0.3 wires the fake tensor-parallel linear layers into the GPT MLP path. With
`tensor_parallel_size=1`, the model keeps the original dense `nn.Linear` MLP.
With `tensor_parallel_size>1`, each transformer MLP uses:

- `ColumnParallelLinear` for the expansion projection from hidden size to the
  4x intermediate size. This shards output features and concatenates the local
  outputs in this single process.
- `GELU` activation.
- `RowParallelLinear` for the projection back to hidden size. This shards input
  features and sums partial outputs before returning to the residual stream.

This remains an educational fake TP implementation. It does not use
`torch.distributed`, NCCL, process groups, rank-local parameters, or real
multi-GPU communication. Benchmark numbers are useful for correctness checks
and intuition only, not for claiming speedup.

Try the MLP comparison and TP training smoke test with:

```bash
python examples/compare_tp_mlp.py
python examples/train_tiny_gpt.py --tensor-parallel-size 2
pytest
```

## v0.4 Fake Tensor Parallel Attention

v0.4 adds fake TP support to GPT attention projections. Attention heads are
sharded across fake TP shards:

- `num_heads` must be divisible by `tensor_parallel_size`.
- Each shard owns `local_heads = num_heads / tensor_parallel_size`.
- QKV rows are sharded by local Q, K, and V heads, not by blindly splitting the
  full `3 * hidden_size` output dimension.
- The attention output projection uses row-parallel-style partial outputs, and
  output bias is applied once.

Embeddings and the LM head are still dense unless implemented separately.

This is still single-process fake tensor parallelism. It does not use
`torch.distributed`, NCCL, process groups, rank-local process state, or real
multi-GPU communication. The examples are for correctness and intuition, not
for speedup claims.

Try the attention comparison with:

```bash
python examples/compare_tp_attention.py
pytest
```

## v0.5 Direction

- Add clearer parameter-count and shard-shape reporting.
- Optionally add a simple pipeline schedule visualization.
