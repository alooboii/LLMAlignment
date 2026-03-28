from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .common import build_next_token_mask, forward_token_log_probs, sequence_log_prob_sum


@dataclass(slots=True)
class DPOOutputs:
    loss: torch.Tensor
    implicit_margin: torch.Tensor
    preference_accuracy: torch.Tensor
    delta_policy: torch.Tensor
    delta_ref: torch.Tensor


def sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    token_log_probs = forward_token_log_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    token_mask = build_next_token_mask(response_mask)
    return sequence_log_prob_sum(token_log_probs, token_mask)


def dpo_loss(
    *,
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> DPOOutputs:
    delta_policy = policy_chosen_logp - policy_rejected_logp
    delta_ref = ref_chosen_logp - ref_rejected_logp
    z = beta * (delta_policy - delta_ref)
    loss = -F.logsigmoid(z).mean()
    preference_accuracy = (delta_policy > 0).float().mean()
    return DPOOutputs(
        loss=loss,
        implicit_margin=z.detach().mean(),
        preference_accuracy=preference_accuracy.detach(),
        delta_policy=delta_policy.detach(),
        delta_ref=delta_ref.detach(),
    )


def dpo_forward_pass(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: dict[str, Any],
    beta: float = 0.1,
) -> DPOOutputs:
    chosen_policy = sequence_log_probs(
        policy_model,
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        response_mask=batch["chosen_response_mask"],
    )
    rejected_policy = sequence_log_probs(
        policy_model,
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        response_mask=batch["rejected_response_mask"],
    )
    with torch.no_grad():
        chosen_ref = sequence_log_probs(
            reference_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            response_mask=batch["chosen_response_mask"],
        )
        rejected_ref = sequence_log_probs(
            reference_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            response_mask=batch["rejected_response_mask"],
        )
    return dpo_loss(
        policy_chosen_logp=chosen_policy,
        policy_rejected_logp=rejected_policy,
        ref_chosen_logp=chosen_ref,
        ref_rejected_logp=rejected_ref,
        beta=beta,
    )

