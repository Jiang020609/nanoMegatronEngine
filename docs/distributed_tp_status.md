# Distributed Tensor Parallel Prototype Status

This document summarizes the current v0.10 distributed tensor-parallel
prototype status. The implementation is a lightweight rewrite of the relevant
Megatron-style tensor-parallel mechanics for learning and validation; it is not
vendored Megatron-LM code and does not claim Megatron-LM feature or performance
parity.

## Scope

The distributed path is an isolated prototype built from rank-local modules:

- `DistributedColumnParallelLinear`
- `DistributedRowParallelLinear`
- `DistributedQKVParallelLinear`
- `VocabParallelEmbedding` with `DistributedRankLocalCollectives`
- `VocabParallelLMHead` with `DistributedRankLocalCollectives`
- `DistributedCausalSelfAttention`
- `DistributedTransformerBlock`
- `DistributedGPTModel`

The main `GPTModel` and trainer path are still unchanged. Fake tensor
parallel behavior remains separate and unchanged.

## Validation Matrix

| Path | Command | What It Checks | Status |
| --- | --- | --- | --- |
| Default tests | `pytest` | Dense path, fake TP path, non-distributed tests, skipped distributed tests by default | Passing locally |
| CPU/Gloo distributed modules | `NME_RUN_DISTRIBUTED_TESTS=1 pytest tests/test_distributed_collectives.py tests/test_distributed_column_parallel_linear.py tests/test_distributed_row_parallel_linear.py tests/test_distributed_vocab_parallel.py tests/test_distributed_mlp_composition.py tests/test_distributed_qkv_parallel_linear.py tests/test_distributed_attention.py tests/test_distributed_transformer_block.py tests/test_distributed_gpt_forward.py tests/test_distributed_dropout_rng.py` | Rank-local collectives, distributed linear/vocab/QKV/attention/block/GPT forward and backward smoke checks, no-bias GPT forward/loss validation, and opt-in tracked-dropout RNG stream semantics | Passing locally |
| CPU/Gloo MLP composition | `python examples/compare_distributed_mlp.py --spawn 2` | Dense MLP vs distributed column-local GELU plus row-parallel MLP, including input gradients | Passing locally |
| CUDA/NCCL GPT smoke | `torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_nccl.py --preset small` | Dense vs distributed GPT forward/loss, backward smoke, replicated gradient sync, local SGD step smoke, activation checkpoint backward smoke | Passed on 4-GPU A800 |
| CUDA/NCCL GPT gradient and optimizer equivalence | `torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_gradients_nccl.py --preset small` | Dense gradients vs local distributed gradient shards, replicated gradients after explicit sync, and one SGD-updated dense parameter slice vs local distributed shard | Passed on 4-GPU A800: 236/236 checks, max abs error around `1.8e-7` |
| CUDA/NCCL GPT multi-step training equivalence | `torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5` | A short deterministic SGD loop comparing logits, loss, gradient shards, replicated gradients, and updated parameter shards after every step | Passed on 4-GPU A800: 1240/1240 checks, max abs error around `4.8e-7` |
| CUDA/NCCL GPT AdamW state equivalence | `torchrun --standalone --nproc_per_node=4 examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5 --optimizer adamw --weight-decay 0.01 --adamw-parameter-atol 1e-4` | A short deterministic AdamW loop comparing logits, loss, gradient shards, replicated gradients, updated parameter shards, and AdamW `step`/`exp_avg`/`exp_avg_sq` state shards after every step. AdamW updated parameter checks use explicit `atol=1e-4`; state/gradient/logit/loss checks use the base tolerance. | Passed on 4-GPU A800: 2920/2920 checks, max abs error around `6.5e-5` |

## Current Guarantees

For the isolated distributed GPT prototype, the validated `--preset small`
CUDA/NCCL path checks:

- full logits match dense GPT within tolerance
- loss matches dense GPT within tolerance
- QKV local head gradient shards match dense Q/K/V head slices
- row-parallel projection and MLP gradient shards match dense column slices
- column-parallel MLP gradient shards match dense row slices
- tied vocab embedding/LM-head gradient shards match dense vocab slices
- replicated parameter gradients match dense after explicit replicated-gradient
  synchronization
- one SGD step updates local distributed shards to match the corresponding
  dense parameter slices
- a 5-step deterministic SGD loop keeps logits, losses, gradient shards,
  replicated gradients, and updated parameter shards aligned with dense slices

On CPU/Gloo, the isolated distributed GPT forward/loss tests cover both the
default projection-bias configuration and `bias=False` for attention and MLP
projection linears. LayerNorm bias and the tied LM head behavior remain
unchanged.

The CUDA/NCCL smoke, gradient-equivalence, and multi-step training diagnostic
scripts accept `--no-bias` to run the same isolated distributed GPT prototype
checks with attention and MLP projection biases disabled. These no-bias CUDA
diagnostics still need an A800 validation run before they are listed as passed.

## Foundational RNG Work

`RNGStateTracker` provides process-local named CPU/CUDA RNG state capture,
restore, and fork semantics. This is intended as the foundation for later
dropout and activation-checkpointing determinism checks. `compare_rng_dropout.py`
checks process-local dropout replay, named-stream advancement, and outer RNG
restoration. The activation checkpoint tests also verify dropout replay for a
checkpointed block against a direct block under the same tracked RNG stream.
`TrackedDropout` can optionally use named RNG streams inside the isolated
distributed attention, transformer block, and GPT prototypes. The opt-in
CPU/Gloo distributed dropout RNG test checks that rank-local streams can differ
across tensor-parallel ranks while replicated streams remain identical across
ranks, and that tracked dropout replays in a distributed transformer block when
stream states are reset. Dropout-on dense-equivalent distributed training is
not claimed yet.

## Latest Strict A800 Check

The latest stricter validation target is a multi-step SGD equivalence run:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5
```

This completed on a 4-GPU A800 environment with 1240/1240 checks passing and
maximum absolute error around `4.8e-7`. It keeps the same dense and distributed
GPT instances alive across multiple deterministic steps. Each step checks
logits, loss, sharded gradients, replicated gradients after explicit
synchronization, and updated parameter shards.

## Latest AdamW A800 Check

The latest stricter AdamW optimizer-state equivalence target is:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5 \
  --optimizer adamw --weight-decay 0.01 --adamw-parameter-atol 1e-4
```

This uses the same short deterministic training loop, but also compares the
AdamW `step`, `exp_avg`, and `exp_avg_sq` state for rank-local shards and
replicated parameters against the corresponding dense parameter or dense tensor
slice. AdamW updated parameter checks use an explicit `atol=1e-4` because the
adaptive update can amplify tiny dense-vs-distributed numerical differences in
bias terms; optimizer state, gradients, logits, and losses still use the base
tolerance. This completed on a 4-GPU A800 environment with 2920/2920 checks
passing and maximum absolute error around `6.5e-5`.

## Important Non-Goals

The current prototype does not implement or claim:

- wiring into the main `GPTModel` path
- a full distributed training engine
- Megatron-LM feature parity
- dropout-on dense-equivalent distributed training
- A800-validated no-bias CUDA/NCCL gradient or training equivalence
- a production distributed optimizer
- mixed precision, FP8, or custom CUDA kernels
- sequence parallelism
- pipeline parallelism
- data parallelism or ZeRO-style optimizer sharding
- multi-node orchestration
- benchmarked speedups or throughput claims

## Useful Commands

Run default tests:

```bash
pytest
```

Run opt-in CPU/Gloo distributed tests:

```bash
NME_RUN_DISTRIBUTED_TESTS=1 pytest \
  tests/test_distributed_collectives.py \
  tests/test_distributed_column_parallel_linear.py \
  tests/test_distributed_row_parallel_linear.py \
  tests/test_distributed_vocab_parallel.py \
  tests/test_distributed_mlp_composition.py \
  tests/test_distributed_qkv_parallel_linear.py \
  tests/test_distributed_attention.py \
  tests/test_distributed_transformer_block.py \
  tests/test_distributed_gpt_forward.py \
  tests/test_distributed_dropout_rng.py
```

Run the CUDA/NCCL strict gradient and optimizer equivalence check:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_gradients_nccl.py --preset small
```

Run the pending no-bias CUDA/NCCL gradient and optimizer equivalence check:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_gradients_nccl.py --preset small --no-bias
```

Run the CUDA/NCCL multi-step training equivalence check:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5
```

Run the pending no-bias CUDA/NCCL multi-step training equivalence check:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5 --no-bias
```

Run the CUDA/NCCL AdamW optimizer-state equivalence check:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/compare_distributed_gpt_training_nccl.py --preset small --steps 5 \
  --optimizer adamw --weight-decay 0.01 --adamw-parameter-atol 1e-4
```

The CUDA/NCCL commands are single-node prototype validations only. They should
not be read as multi-node, production-training, or speedup claims.

Run the process-local RNG/dropout determinism check:

```bash
python examples/compare_rng_dropout.py --device cpu
```
