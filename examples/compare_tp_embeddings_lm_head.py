"""Compare dense vocab layers with fake vocab-parallel layers."""

from __future__ import annotations

import torch
from torch import nn

from nano_megatron_engine.parallel import VocabParallelEmbedding, VocabParallelLMHead


def main() -> None:
    torch.manual_seed(2029)
    vocab_size = 65
    hidden_size = 16
    tensor_parallel_size = 2

    dense_embedding = nn.Embedding(vocab_size, hidden_size)
    dense_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    dense_lm_head.weight = dense_embedding.weight

    tp_embedding = VocabParallelEmbedding.from_embedding(dense_embedding, tp_size=tensor_parallel_size)
    tp_lm_head = VocabParallelLMHead(hidden_size, vocab_size, tp_size=tensor_parallel_size, bias=False)
    tp_lm_head.tie_weight_shards(tp_embedding.weight_shards)

    input_ids = torch.tensor([[0, 1, 32, 33], [34, 48, 63, 64]])
    hidden_states = torch.randn(2, 4, hidden_size)

    dense_embedding_output = dense_embedding(input_ids)
    tp_embedding_output = tp_embedding(input_ids)
    dense_logits = dense_lm_head(hidden_states)
    tp_logits = tp_lm_head(hidden_states)

    embedding_error = (tp_embedding_output - dense_embedding_output).abs().max().item()
    logits_error = (tp_logits - dense_logits).abs().max().item()

    print("Fake TP embedding and LM-head comparison")
    print("This is single-process educational fake TP, not distributed communication.")
    print(f"vocab_size={vocab_size}")
    print(f"hidden_size={hidden_size}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    for shard_idx, (start, end) in enumerate(tp_embedding.vocab_ranges):
        print(f"shard={shard_idx} vocab_range=[{start}, {end}) local_vocab_size={end - start}")
    print(f"dense_embedding_shape={tuple(dense_embedding_output.shape)}")
    print(f"tp_embedding_shape={tuple(tp_embedding_output.shape)}")
    print(f"dense_logits_shape={tuple(dense_logits.shape)}")
    print(f"tp_logits_shape={tuple(tp_logits.shape)}")
    print(f"embedding_max_abs_error={embedding_error:.3e}")
    print(f"logits_max_abs_error={logits_error:.3e}")
    print(f"embedding_outputs_close={torch.allclose(tp_embedding_output, dense_embedding_output, atol=1e-6, rtol=1e-5)}")
    print(f"logits_close={torch.allclose(tp_logits, dense_logits, atol=1e-6, rtol=1e-5)}")


if __name__ == "__main__":
    main()
