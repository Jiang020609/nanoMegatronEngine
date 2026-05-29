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

## Capability Table

| Area | Current support | Limitations |
| --- | --- | --- |
| Dense GPT training | CPU-runnable tiny GPT, trainer, microbatching, gradient accumulation, activation checkpointing | Educational scale only |
| Fake tensor parallel layers | Single-process MLP, attention, embeddings, LM head, and fake collectives | Local tensors only, no process groups |
| Distributed collectives | Planned for v0.7 as optional CPU/Gloo wrappers in `distributed_collectives.py` | Not used by GPT TP yet; normal pytest must not require distributed setup |
| Accelerators | CUDA is optional for existing benchmarks | No NCCL, custom CUDA, FP8, or GPU requirement |
| Performance claims | None | No Megatron-LM parity or speedup claims |

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
python examples/compare_tp_embeddings_lm_head.py
python examples/inspect_fake_collectives.py
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

This is still single-process fake tensor parallelism. It does not use
`torch.distributed`, NCCL, process groups, rank-local process state, or real
multi-GPU communication. The examples are for correctness and intuition, not
for speedup claims.

Try the attention comparison with:

```bash
python examples/compare_tp_attention.py
pytest
```

## v0.5 Fake Vocab Parallel Embeddings And LM Head

v0.5 adds fake TP support to token embeddings and the LM head:

- Vocab-parallel token embeddings split vocab rows across fake shards using
  contiguous ranges. Uneven vocab sizes are supported, so `vocab_size=65` and
  `tensor_parallel_size=2` becomes `[0, 33)` and `[33, 65)`.
- Each embedding shard only handles token ids in its local vocab range. Outputs
  from all shards are summed back into the usual `[batch, seq, hidden_size]`
  tensor.
- The vocab-parallel LM head splits the output vocab dimension across fake
  shards, computes local logits, then gathers them back into full
  `[batch, seq, vocab_size]` logits.
- The existing GPT embedding and LM-head weight tying is preserved by sharing
  vocab shard parameters.

v0.3 added fake TP MLP layers, v0.4 added fake TP attention projections, and
v0.5 extends that teaching path to the vocabulary-facing layers.

This remains single-process fake tensor parallelism. It does not use
`torch.distributed`, NCCL, process groups, rank-local process state, or real
multi-GPU communication, and it does not claim speedups.

Try the vocab comparison with:

```bash
python examples/compare_tp_embeddings_lm_head.py
pytest
```

## v0.6 Fake Collective APIs

v0.6 introduces explicit fake tensor-parallel collective APIs in `fake_tp.py`:

- `fake_all_gather` simulates gathering ordered shard outputs by concatenating
  tensors along a chosen dimension. It supports uneven sizes along the gather
  dimension.
- `fake_all_reduce_sum` simulates summing partial outputs across fake shards.
- `fake_reduce_scatter_sum` simulates a sum followed by returning one
  contiguous output partition.
- `partition_range` remains the helper for contiguous, uneven partitioning.

The fake TP layers now call these APIs where it clarifies the Megatron-style
communication pattern:

- `ColumnParallelLinear` uses `fake_all_gather` when `gather_output=True`.
- `RowParallelLinear` uses `fake_all_reduce_sum`.
- `VocabParallelEmbedding` uses `fake_all_reduce_sum`.
- `VocabParallelLMHead` uses `fake_all_gather`.

This is still single-process fake TP. It does not use `torch.distributed`,
NCCL, process groups, rank-local process state, or real multi-GPU
communication, and it does not claim speedups.

Inspect the fake collectives with:

```bash
python examples/inspect_fake_collectives.py
pytest
```

## v0.7 Direction

- Add clearer parameter-count and shard-shape reporting.
- Optionally add a simple pipeline schedule visualization.
