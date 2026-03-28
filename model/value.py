from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from .loading import load_backbone_model


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Backbone config must expose hidden_size.")
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name: str,
        dtype: str = "bfloat16",
        load_in_bits: int | None = None,
        device_map: str | dict[str, Any] = "auto",
    ) -> "ValueModel":
        backbone = load_backbone_model(
            model_name=model_name,
            dtype=dtype,
            load_in_bits=load_in_bits,
            device_map=device_map,
        )
        return cls(backbone=backbone)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        values = self.value_head(hidden).squeeze(-1)
        return values

