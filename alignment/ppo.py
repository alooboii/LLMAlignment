from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .common import (
    build_next_token_mask,
    forward_token_log_probs,
    masked_mean,
    masked_std,
    token_log_probs_from_logits,
)


@dataclass(slots=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    kl_loss_coef: float = 0.0


@dataclass(slots=True)
class PPOLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl_old_new: torch.Tensor
    approx_kl_ref: torch.Tensor
    mean_ratio: torch.Tensor


def build_token_rewards(
    *,
    task_rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_token_mask: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """Per-token rewards with terminal task reward + dense KL penalty."""
    rewards = -kl_coef * (old_log_probs - ref_log_probs)
    rewards = rewards * response_token_mask.float()

    batch_size, width = rewards.shape
    for i in range(batch_size):
        valid = torch.nonzero(response_token_mask[i], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        rewards[i, valid[-1]] += task_rewards[i]
    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_token_mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GAE on token trajectories.

    Args:
        rewards: [B, T]
        values: [B, T]
        response_token_mask: [B, T] bool
    """
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    width = rewards.shape[1]

    for t in range(width - 1, -1, -1):
        mask_t = response_token_mask[:, t].float()
        if t + 1 < width:
            next_value = values[:, t + 1]
            next_mask = response_token_mask[:, t + 1].float()
        else:
            next_value = torch.zeros_like(last_adv)
            next_mask = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last_adv = delta + gamma * gae_lambda * last_adv * next_mask
        advantages[:, t] = last_adv * mask_t
        last_adv = last_adv * mask_t

    returns = (advantages + values) * response_token_mask.float()
    return advantages, returns


def standardize_advantages(
    advantages: torch.Tensor,
    response_token_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mu = masked_mean(advantages, response_token_mask, eps=eps)
    sigma = masked_std(advantages, response_token_mask, eps=eps)
    standardized = (advantages - mu) / (sigma + eps)
    return standardized * response_token_mask.float()


def ppo_losses(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    response_token_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    kl_loss_coef: float = 0.0,
    token_entropies: torch.Tensor | None = None,
) -> PPOLossOutput:
    mask = response_token_mask.bool()
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_objective = torch.minimum(unclipped, clipped)
    policy_loss = -masked_mean(policy_objective, mask)

    value_loss = masked_mean((new_values - returns.detach()).square(), mask)
    if token_entropies is None:
        entropy = new_log_probs.new_tensor(0.0)
    else:
        entropy = masked_mean(token_entropies, mask)

    approx_kl_old_new = masked_mean(old_log_probs - new_log_probs, mask)
    approx_kl_ref = masked_mean(new_log_probs - ref_log_probs, mask)
    total = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy
        + kl_loss_coef * approx_kl_ref
    )
    return PPOLossOutput(
        total_loss=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl_old_new=approx_kl_old_new.detach(),
        approx_kl_ref=approx_kl_ref.detach(),
        mean_ratio=masked_mean(ratio, mask).detach(),
    )


def compute_token_log_probs_and_entropy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = token_log_probs_from_logits(logits, labels)

    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return log_probs, entropy


def gae_unit_test(tol: float = 1e-6) -> bool:
    rewards = torch.tensor([[0.05, -0.02, 1.6]], dtype=torch.float32)
    values = torch.tensor([[1.5, 1.55, 1.45]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    adv, _ = compute_gae(rewards, values, mask, gamma=1.0, gae_lambda=1.0)
    expected = torch.tensor([[0.13, 0.03, 0.15]], dtype=torch.float32)
    return bool(torch.allclose(adv, expected, atol=tol))


def ratio_sanity_test(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, tol: float = 1e-5) -> bool:
    ratio = torch.exp(new_log_probs - old_log_probs)
    return bool((ratio - 1.0).abs().max().item() < tol)


def clipping_sanity_test(epsilon: float = 0.2, tol: float = 1e-6) -> bool:
    log_ratio = torch.tensor([torch.log(torch.tensor(1.5))], requires_grad=True)
    ratio = torch.exp(log_ratio)
    advantage = torch.tensor([1.0])
    objective = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
    loss = -objective.mean()
    loss.backward()
    clipped_value_ok = abs(objective.item() - (1 + epsilon)) < tol
    grad_ok = abs(log_ratio.grad.item()) < tol
    return clipped_value_ok and grad_ok

