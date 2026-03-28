from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class RewardLossOutput:
    loss: torch.Tensor
    pair_loss: torch.Tensor
    reg_loss: torch.Tensor
    preference_accuracy: torch.Tensor


def _last_non_pad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=-1).clamp(min=1)
    return lengths - 1


def reward_scores(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    if logits.ndim == 2:  # [B, 1] for sequence classification
        return logits.squeeze(-1)
    if logits.ndim == 3:  # [B, T, 1] fallback
        idx = _last_non_pad_indices(attention_mask)
        gathered = logits[torch.arange(logits.size(0), device=logits.device), idx]
        return gathered.squeeze(-1)
    raise ValueError(f"Unexpected reward logits shape: {tuple(logits.shape)}")


def preference_accuracy_from_rewards(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    return (chosen > rejected).float().mean()


def pairwise_reward_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    reg_lambda: float = 1e-3,
) -> RewardLossOutput:
    chosen_scores = reward_scores(
        model=model,
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    )
    rejected_scores = reward_scores(
        model=model,
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    )

    pair_loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
    reg_loss = (chosen_scores.square().mean() + rejected_scores.square().mean()) * reg_lambda
    total = pair_loss + reg_loss
    accuracy = preference_accuracy_from_rewards(chosen_scores, rejected_scores)
    return RewardLossOutput(
        loss=total,
        pair_loss=pair_loss,
        reg_loss=reg_loss,
        preference_accuracy=accuracy,
    )

