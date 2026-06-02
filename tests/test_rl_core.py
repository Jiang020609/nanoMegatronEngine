import math

import pytest
import torch
from torch.nn import functional as F

from nano_megatron_engine.rl import (
    apply_kl_penalty,
    entropy_bonus,
    generalized_advantage_estimate,
    masked_kl_divergence,
    masked_mean,
    next_token_logprobs,
    ppo_clipped_policy_loss,
    response_action_mask,
    terminal_reward_to_token_rewards,
    value_loss,
)


def test_response_action_mask_marks_response_next_tokens():
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
    )
    prompt_lengths = torch.tensor([2, 4])

    mask = response_action_mask(tokens, prompt_lengths)

    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, False, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_next_token_logprobs_matches_manual_gather():
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
                [1.0, 2.0, 0.0],
            ]
        ]
    )
    tokens = torch.tensor([[0, 2, 1]])

    actual = next_token_logprobs(logits, tokens)
    expected = F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_masked_kl_and_rewards():
    mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    scores = torch.tensor([2.0, 3.0])
    rewards = terminal_reward_to_token_rewards(scores, mask)
    expected_rewards = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    torch.testing.assert_close(rewards, expected_rewards)

    kl = torch.full_like(rewards, 0.5)
    penalized = apply_kl_penalty(rewards, kl, mask, kl_coef=0.2)
    torch.testing.assert_close(penalized[mask], rewards[mask] - 0.1)
    torch.testing.assert_close(penalized[~mask], torch.zeros_like(penalized[~mask]))

    policy_logprobs = torch.tensor([[0.0, 0.2, 0.4, 0.0], [0.0, 0.0, 0.3, 0.5]])
    reference_logprobs = torch.zeros_like(policy_logprobs)
    expected_kl = policy_logprobs[mask].mean()
    torch.testing.assert_close(masked_kl_divergence(policy_logprobs, reference_logprobs, mask), expected_kl)


def test_generalized_advantage_estimate_masks_padded_actions():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 9.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        mask,
        gamma=1.0,
        lam=1.0,
    )

    expected = torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_ppo_value_and_entropy_losses():
    old_logprobs = torch.zeros(1, 2)
    new_logprobs = torch.log(torch.tensor([[1.1, 1.5]]))
    advantages = torch.ones(1, 2)
    mask = torch.tensor([[True, True]])

    policy_loss, approx_kl = ppo_clipped_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        clip_epsilon=0.2,
    )

    torch.testing.assert_close(policy_loss, torch.tensor(-1.15))
    torch.testing.assert_close(approx_kl, (old_logprobs - new_logprobs).mean())

    values = torch.tensor([[1.0, 3.0]])
    returns = torch.tensor([[2.0, 1.0]])
    torch.testing.assert_close(value_loss(values, returns, mask), torch.tensor(1.25))

    logits = torch.zeros(1, 2, 2)
    torch.testing.assert_close(entropy_bonus(logits, mask), torch.tensor(math.log(2.0)))


def test_rl_helpers_validate_shapes():
    with pytest.raises(ValueError, match="sequence length"):
        response_action_mask(torch.ones(2, 1, dtype=torch.long), torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match="same shape"):
        masked_mean(torch.ones(2, 2), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="clip_epsilon"):
        ppo_clipped_policy_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), clip_epsilon=-1.0)
