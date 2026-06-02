# nanoMegatronEngine Roadmap

This roadmap tracks the long-term goal of turning nanoMegatronEngine into a
lightweight learning implementation of Megatron-style distributed training
mechanics plus a small RL/RLHF infrastructure track. The project should remain
readable, testable, and correctness-first. It should not vendor Megatron-LM
code or claim Megatron-LM feature or performance parity.

## Current Baseline

The repository currently has three distinct paths:

- Dense tiny GPT training: the default `GPTModel`, trainer utilities,
  microbatching, gradient accumulation, activation checkpointing, and tests.
- Fake tensor parallelism: single-process educational TP modules and
  collectives that preserve the default GPT path and do not use
  `torch.distributed`.
- Isolated distributed tensor-parallel prototype: rank-local modules and
  examples that compare against dense GPT on CPU/Gloo and CUDA/NCCL.

The current distributed TP validation status is summarized in
[`distributed_tp_status.md`](distributed_tp_status.md).

## Guiding Principles

- Keep fake TP, dense GPT, and real distributed prototype boundaries explicit.
- Prefer small, inspectable modules over production abstractions.
- Add real checks before claiming support for a new distributed behavior.
- Keep default `pytest` runnable without process group initialization.
- Treat CUDA/NCCL examples as single-node prototype validation only.
- Avoid speedup, multi-node, and Megatron-LM parity claims.

## Track A: Megatron-Style Distributed Training Mechanics

### A1. Tensor Parallel Completeness

Status: in progress.

Implemented so far:

- distributed column-parallel linear
- distributed row-parallel linear
- distributed QKV head-sharded projection
- distributed vocab embedding and LM head
- distributed causal self-attention
- distributed transformer block
- isolated distributed GPT model
- dense-vs-distributed forward, gradient, SGD, and AdamW state validation
- process-local named CPU/CUDA RNG state tracker
- process-local dropout determinism check using the RNG tracker
- activation checkpointing plus dropout replay test under a tracked RNG stream
- optional tracked dropout in isolated distributed prototypes
- CPU/Gloo rank-local vs replicated dropout RNG stream checks
- GPT projection bias configuration
- CPU/Gloo no-bias distributed GPT forward/loss validation

Next targets:

- optional explicit factory for distributed GPT construction
- no-bias CUDA/NCCL gradient and training diagnostics

### A2. Data Parallelism

Status: not implemented.

Planned work:

- single-node process-group layout for TP and DP groups
- DP-only gradient all-reduce smoke checks
- TP+DP rank layout metadata
- dense batch vs DP split-batch gradient equivalence
- documentation of replicated vs sharded parameter behavior

Non-goals for the first pass:

- ZeRO-style optimizer sharding
- multi-node orchestration
- production DDP replacement

### A3. Pipeline Parallelism

Status: not implemented.

Planned work:

- simple pipeline schedule visualization
- GPipe-style microbatch timeline
- toy stage partitioning
- optional CPU/Gloo send/recv prototype
- basic activation storage and bubble accounting

Non-goals for the first pass:

- interleaved pipeline schedules
- multi-node runtime
- performance claims

### A4. Sequence Parallelism

Status: not implemented.

Planned work:

- sequence-shard tensor helpers
- LayerNorm sequence-parallel semantics
- reduce-scatter/all-gather checks around TP modules
- interaction with dropout RNG tracking
- CPU/Gloo tests before CUDA/NCCL examples

### A5. Distributed Optimizer / ZeRO-Ish Learning Path

Status: partially diagnosed, not implemented as a production optimizer.

Implemented so far:

- dense-vs-distributed local AdamW state equivalence checks for the isolated
  distributed GPT prototype

Planned work:

- optimizer state partitioning utilities
- local state update equivalence tests
- checkpoint gather/scatter helpers
- explicit no-production-optimizer documentation

## Track B: RL / RLHF Infrastructure

Status: not implemented.

The RL track should start as a dense, single-process training objective layer
and only later connect to distributed TP. This keeps RL correctness separate
from distributed correctness while both are still small.

### B1. RL Core Utilities

Planned package:

```text
nano_megatron_engine/rl/
  __init__.py
  data.py
  logprobs.py
  advantages.py
  losses.py
  rewards.py
  rollout.py
  ppo.py
```

Planned work:

- prompt/response batch data structures
- action masks for response tokens
- token log-prob extraction
- reference-model KL helpers
- reward shaping utilities
- generalized advantage estimation
- PPO clipped policy loss
- value loss
- entropy bonus
- toy reward functions

### B2. Dense Tiny PPO Loop

Planned work:

- policy model: dense `GPTModel`
- reference model: frozen dense `GPTModel`
- value head wrapper
- synthetic prompt/response generation
- toy reward model
- one-step and multi-step PPO examples
- tests proving reference gradients stay frozen

### B3. RL + Distributed TP

Planned work:

- distributed policy log-prob equivalence
- dense PPO loss vs distributed PPO loss
- PPO gradient shard equivalence
- AdamW state equivalence under PPO loss
- optional distributed reference model

This is the point where the distributed systems track and RL track should
meet.

## Recommended Near-Term Order

1. Add an optional explicit factory for distributed GPT construction.
2. Add no-bias CUDA/NCCL gradient and training diagnostics.
3. Add dropout-on dense-vs-distributed diagnostics using the tracked streams.
4. Start RL core utilities with log-probs, masks, GAE, and PPO losses.
5. Add a dense tiny PPO example.

This order keeps the already-validated TP path stable while preparing the
randomness and objective-layer semantics needed for realistic training.
