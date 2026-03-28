from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable, Sequence

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


def build_lora_config(
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    task_type: str = "causal_lm",
) -> LoraConfig:
    task_map = {
        "causal_lm": TaskType.CAUSAL_LM,
        "sequence_classification": TaskType.SEQ_CLS,
        "seq_cls": TaskType.SEQ_CLS,
        "feature_extraction": TaskType.FEATURE_EXTRACTION,
    }
    resolved = task_map.get(task_type.lower(), TaskType.CAUSAL_LM)
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        task_type=resolved,
        bias="none",
    )


def apply_lora(model: torch.nn.Module, lora_config: LoraConfig) -> PeftModel:
    peft_model = get_peft_model(model, lora_config)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    return peft_model


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def clone_and_freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    frozen = copy.deepcopy(model)
    return freeze_model(frozen)


@contextmanager
def temporary_disable_adapters(model: torch.nn.Module):
    if hasattr(model, "disable_adapter") and callable(model.disable_adapter):
        with model.disable_adapter():
            yield
        return

    # PEFT fallback API (older versions).
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            yield
        finally:
            model.enable_adapter_layers()
        return

    with nullcontext():
        yield


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)

