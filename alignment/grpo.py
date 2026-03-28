from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import masked_std


@dataclass(slots=True)
class GRPOConfig:
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    kl_loss_coef: float = 0.0


@dataclass(slots=True)
class GRPOLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    approx_kl_ref: torch.Tensor
    mean_ratio: torch.Tensor
    degenerate_fraction: float


def group_relative_advantages(group_rewards: torch.Tensor) -> torch.Tensor:
    """group_rewards shape: [B, K]."""
    group_means = group_rewards.mean(dim=1, keepdim=True)
    return group_rewards - group_means


def degenerate_group_fraction(group_rewards: torch.Tensor, tol: float = 1e-12) -> float:
    # A group is degenerate when all K rewards are effectively equal.
    spread = group_rewards.max(dim=1).values - group_rewards.min(dim=1).values
    return float((spread < tol).float().mean().item())


def broadcast_group_advantages_to_tokens(
    *,
    group_advantages: torch.Tensor,
    response_token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        group_advantages: [B, K]
        response_token_mask: [B*K, T]
    Returns:
        token_advantages: [B*K, T]
    """
    batch_groups, k = group_advantages.shape
    flat_adv = group_advantages.reshape(batch_groups * k, 1)
    return flat_adv * response_token_mask.float()


def standardize_token_advantages(token_advantages: torch.Tensor, response_token_mask: torch.Tensor) -> torch.Tensor:
    mu = masked_mean(token_advantages, response_token_mask)
    sigma = masked_std(token_advantages, response_token_mask)
    return (token_advantages - mu) / (sigma + 1e-8) * response_token_mask.float()


def grpo_policy_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    token_advantages: torch.Tensor,
    response_token_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_loss_coef: float = 0.0,
    group_rewards: torch.Tensor | None = None,
) -> GRPOLossOutput:
    mask = response_token_mask.bool()
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages
    token_objective = torch.minimum(unclipped, clipped)
    mask_f = mask.float()
    seq_len = mask_f.sum(dim=1).clamp(min=1.0)
    seq_objective = (token_objective * mask_f).sum(dim=1) / seq_len
    policy_loss = -seq_objective.mean()

    seq_kl = ((new_log_probs - ref_log_probs) * mask_f).sum(dim=1) / seq_len
    approx_kl_ref = seq_kl.mean()
    total_loss = policy_loss + kl_loss_coef * approx_kl_ref

    degenerate = degenerate_group_fraction(group_rewards) if group_rewards is not None else 0.0
    seq_ratio = (ratio * mask_f).sum(dim=1) / seq_len
    return GRPOLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        approx_kl_ref=approx_kl_ref.detach(),
        mean_ratio=seq_ratio.mean().detach(),
        degenerate_fraction=degenerate,
    )
