from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from data.gsm8k import extract_numeric_answer, verifiable_reward


def build_rlvr_reward_fn(gold_answers: Sequence[str | None]) -> Callable[[Sequence[str]], torch.Tensor]:
    def _reward_fn(predicted_solutions: Sequence[str]) -> torch.Tensor:
        if len(predicted_solutions) != len(gold_answers):
            raise ValueError(
                f"Predictions length ({len(predicted_solutions)}) must match gold length ({len(gold_answers)})."
            )
        rewards = [
            verifiable_reward(pred, gold)
            for pred, gold in zip(predicted_solutions, gold_answers, strict=True)
        ]
        return torch.tensor(rewards, dtype=torch.float32)

    return _reward_fn


def format_compliance_rate(predicted_solutions: Sequence[str]) -> float:
    if not predicted_solutions:
        return 0.0
    with_number = sum(extract_numeric_answer(text) is not None for text in predicted_solutions)
    return with_number / len(predicted_solutions)

