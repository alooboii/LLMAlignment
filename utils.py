from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def dataloader_cycle(loader: Any) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


@dataclass(slots=True)
class Timer:
    start_time: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

