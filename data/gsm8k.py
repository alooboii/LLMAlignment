from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional at import time
    HFDataset = Any  # type: ignore[misc, assignment]
    load_dataset = None


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _canonicalize_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        dec = Decimal(cleaned)
    except InvalidOperation:
        return None
    # Normalize (e.g., "2.0" -> "2"), but preserve sign for negatives.
    normalized = format(dec.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def extract_numeric_answer(text: str) -> str | None:
    candidates: list[str] = []

    patterns = [
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"[Tt]he answer is\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"boxed\{([-+]?\d[\d,]*(?:\.\d+)?)\}",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text):
            candidates.append(match.group(1))

    if not candidates:
        all_numbers = _NUMBER_RE.findall(text)
        if all_numbers:
            candidates.append(all_numbers[-1])

    for raw in reversed(candidates):
        canonical = _canonicalize_number(raw)
        if canonical is not None:
            return canonical
    return None


def extract_gold_answer(gsm8k_solution: str) -> str | None:
    return extract_numeric_answer(gsm8k_solution)


def verifiable_reward(prediction_text: str, gold_answer: str | None) -> float:
    if gold_answer is None:
        return 0.0
    pred = extract_numeric_answer(prediction_text)
    return 1.0 if pred is not None and pred == gold_answer else 0.0


def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step.\n"
        "At the end, write your final answer as a single number.\n"
        f"Problem: {question.strip()}\n"
        "Solution:"
    )


def load_gsm8k(
    split: str,
    dataset_name: str = "openai/gsm8k",
    config: str = "main",
) -> HFDataset:
    if load_dataset is None:
        raise ImportError("datasets is required. Install with `pip install datasets`.")
    return load_dataset(dataset_name, config, split=split)

