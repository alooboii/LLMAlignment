from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional
    PeftModel = None  # type: ignore[assignment]


@dataclass(slots=True)
class ModelStats:
    total_params: int
    trainable_params: int
    trainable_ratio: float


def count_parameters(model: torch.nn.Module) -> ModelStats:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total) if total > 0 else 0.0
    return ModelStats(total_params=total, trainable_params=trainable, trainable_ratio=ratio)


def describe_cuda_memory(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return f"{prefix}cuda_unavailable"
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    return f"{prefix}allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB"


def _quantization_config(load_in_bits: int | None) -> Any:
    if load_in_bits not in (4, 8):
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "bitsandbytes/transformers BitsAndBytesConfig not available. "
            "Install `bitsandbytes` for 4-bit or 8-bit loading."
        )
    return BitsAndBytesConfig(
        load_in_4bit=load_in_bits == 4,
        load_in_8bit=load_in_bits == 8,
    )


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(dtype.lower(), torch.bfloat16)


def load_causal_lm(
    model_name: str,
    dtype: str = "bfloat16",
    load_in_bits: int | None = None,
    device_map: str | dict[str, Any] = "auto",
) -> AutoModelForCausalLM:
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if load_in_bits in (4, 8):
        kwargs["quantization_config"] = _quantization_config(load_in_bits)
    else:
        kwargs["torch_dtype"] = _resolve_dtype(dtype)
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_backbone_model(
    model_name: str,
    dtype: str = "bfloat16",
    load_in_bits: int | None = None,
    device_map: str | dict[str, Any] = "auto",
) -> AutoModel:
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if load_in_bits in (4, 8):
        kwargs["quantization_config"] = _quantization_config(load_in_bits)
    else:
        kwargs["torch_dtype"] = _resolve_dtype(dtype)
    return AutoModel.from_pretrained(model_name, **kwargs)


def load_policy_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    load_in_bits: int | None = None,
    device_map: str | dict[str, Any] = "auto",
    enable_gradient_checkpointing: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_causal_lm(
        model_name=model_name,
        dtype=dtype,
        load_in_bits=load_in_bits,
        device_map=device_map,
    )
    model.config.use_cache = False
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def load_policy_or_adapter(
    model_or_adapter_path: str,
    dtype: str = "bfloat16",
    load_in_bits: int | None = None,
    device_map: str | dict[str, Any] = "auto",
    is_trainable_adapter: bool = True,
    enable_gradient_checkpointing: bool = True,
):
    adapter_cfg_path = Path(model_or_adapter_path) / "adapter_config.json"
    if adapter_cfg_path.exists():
        if PeftModel is None:
            raise ImportError("peft is required to load adapter checkpoints.")
        with adapter_cfg_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = load_causal_lm(
            model_name=base_name,
            dtype=dtype,
            load_in_bits=load_in_bits,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, model_or_adapter_path, is_trainable=is_trainable_adapter)
        model.config.use_cache = False
        if enable_gradient_checkpointing and is_trainable_adapter:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        return model, tokenizer

    return load_policy_and_tokenizer(
        model_name=model_or_adapter_path,
        dtype=dtype,
        load_in_bits=load_in_bits,
        device_map=device_map,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )


def load_reward_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    load_in_bits: int | None = None,
    device_map: str | dict[str, Any] = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "num_labels": 1,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if load_in_bits in (4, 8):
        kwargs["quantization_config"] = _quantization_config(load_in_bits)
    else:
        kwargs["torch_dtype"] = _resolve_dtype(dtype)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
