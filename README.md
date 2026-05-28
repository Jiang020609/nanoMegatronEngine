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
```

## v0.2 Direction

- Add fake tensor parallel `ColumnParallelLinear` and `RowParallelLinear`.
- Add tests comparing fake tensor-parallel layers to `torch.nn.Linear`.
- Add a simple pipeline schedule visualization.
