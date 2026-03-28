from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=eps)
    return (values * mask_f).sum() / denom


def masked_std(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = masked_mean(values, mask, eps=eps)
    centered = (values - mu) * mask.float()
    var = centered.square().sum() / mask.float().sum().clamp(min=eps)
    return torch.sqrt(var + eps)


def token_log_probs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def forward_token_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    return token_log_probs_from_logits(logits, labels)


def build_next_token_mask(response_mask: torch.Tensor) -> torch.Tensor:
    # response_mask marks token positions in input_ids. For next-token log-probs we shift by 1.
    if response_mask.size(1) < 2:
        return response_mask.new_zeros((response_mask.size(0), 0), dtype=torch.bool)
    return response_mask[:, 1:].bool()


def sequence_log_prob_sum(token_log_probs: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    return (token_log_probs * token_mask.float()).sum(dim=-1)

