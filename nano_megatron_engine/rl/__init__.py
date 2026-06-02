"""Small RL/RLHF utility layer for nanoMegatronEngine."""

from nano_megatron_engine.rl.advantages import generalized_advantage_estimate
from nano_megatron_engine.rl.data import masked_mean, response_action_mask
from nano_megatron_engine.rl.logprobs import masked_kl_divergence, next_token_logprobs
from nano_megatron_engine.rl.losses import entropy_bonus, ppo_clipped_policy_loss, value_loss
from nano_megatron_engine.rl.rewards import apply_kl_penalty, terminal_reward_to_token_rewards

__all__ = [
    "apply_kl_penalty",
    "entropy_bonus",
    "generalized_advantage_estimate",
    "masked_kl_divergence",
    "masked_mean",
    "next_token_logprobs",
    "ppo_clipped_policy_loss",
    "response_action_mask",
    "terminal_reward_to_token_rewards",
    "value_loss",
]
