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
python examples/bench_microbatch.py
python examples/bench_parallel_linear.py
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

## v0.3 Direction

- Add fake tensor parallel `ColumnParallelLinear` and `RowParallelLinear`.
- Integrate fake TP layers into the GPT MLP.
- Add a config flag such as `tensor_parallel_size`.
- Optionally add a simple pipeline schedule visualization.
