from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional at import time
    HFDataset = Any  # type: ignore[misc, assignment]
    load_dataset = None


_ASSISTANT_TAGS: tuple[str, ...] = ("\n\nAssistant:", "\nAssistant:", "Assistant:")


@dataclass(slots=True)
class PreferenceTriple:
    prompt: str
    chosen: str
    rejected: str


class PreferenceTripleDataset(Dataset[PreferenceTriple]):
    def __init__(self, triples: Sequence[PreferenceTriple]):
        self._triples = list(triples)

    def __len__(self) -> int:
        return len(self._triples)

    def __getitem__(self, idx: int) -> PreferenceTriple:
        return self._triples[idx]


def _longest_common_prefix(a: str, b: str) -> str:
    max_len = min(len(a), len(b))
    i = 0
    while i < max_len and a[i] == b[i]:
        i += 1
    return a[:i]


def split_prompt_response(conversation: str) -> tuple[str, str]:
    text = conversation.replace("\r\n", "\n").strip()
    best_idx = -1
    best_tag = ""
    for tag in _ASSISTANT_TAGS:
        idx = text.rfind(tag)
        if idx > best_idx:
            best_idx = idx
            best_tag = tag
    if best_idx == -1:
        # Fallback for malformed rows.
        pivot = text.rfind("\n")
        if pivot == -1:
            return "", text
        return text[:pivot].strip(), text[pivot + 1 :].strip()

    split_point = best_idx + len(best_tag)
    prompt = text[:split_point]
    response = text[split_point:].strip()
    return prompt, response


def parse_preference_example(example: dict[str, Any]) -> PreferenceTriple:
    chosen_prompt, chosen_response = split_prompt_response(example["chosen"])
    rejected_prompt, rejected_response = split_prompt_response(example["rejected"])
    if chosen_prompt != rejected_prompt:
        prompt = _longest_common_prefix(chosen_prompt, rejected_prompt).rstrip()
        if "Assistant:" in chosen_prompt:
            # Keep the prompt ending at Assistant tag when possible.
            cutoff = prompt.rfind("Assistant:")
            if cutoff != -1:
                prompt = prompt[: cutoff + len("Assistant:")]
        if not prompt:
            prompt = chosen_prompt or rejected_prompt
    else:
        prompt = chosen_prompt

    return PreferenceTriple(prompt=prompt, chosen=chosen_response, rejected=rejected_response)


def load_hh_harmless(
    split: str,
    dataset_name: str = "Anthropic/hh-rlhf",
    harmless_config: str = "harmless-base",
) -> HFDataset:
    if load_dataset is None:
        raise ImportError("datasets is required. Install with `pip install datasets`.")

    # Primary attempt: load harmless config directly.
    try:
        return load_dataset(dataset_name, harmless_config, split=split)
    except Exception:
        pass

    # Fallback 1: some loaders expose subset via data_dir.
    try:
        return load_dataset(dataset_name, data_dir=harmless_config, split=split)
    except Exception:
        pass

    # Fallback 2 (strict harmless): load harmless files explicitly.
    # This avoids silently loading all HH subsets (~160k) when config routing changes.
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split for harmless subset: {split}")
    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/harmless-base/{split}.jsonl.gz"
    return load_dataset("json", data_files={split: url}, split=split)


def build_hh_datasets(
    train_limit: int | None = None,
    test_limit: int | None = None,
    dataset_name: str = "Anthropic/hh-rlhf",
    harmless_config: str = "harmless-base",
) -> tuple[PreferenceTripleDataset, PreferenceTripleDataset]:
    raw_train = load_hh_harmless(
        split="train",
        dataset_name=dataset_name,
        harmless_config=harmless_config,
    )
    raw_test = load_hh_harmless(
        split="test",
        dataset_name=dataset_name,
        harmless_config=harmless_config,
    )

    train_rows = raw_train if train_limit is None else raw_train.select(range(min(train_limit, len(raw_train))))
    test_rows = raw_test if test_limit is None else raw_test.select(range(min(test_limit, len(raw_test))))

    train_triples = [parse_preference_example(row) for row in train_rows]
    test_triples = [parse_preference_example(row) for row in test_rows]
    return PreferenceTripleDataset(train_triples), PreferenceTripleDataset(test_triples)


def print_sample_triples(dataset: Sequence[PreferenceTriple], k: int = 3) -> None:
    for i in range(min(k, len(dataset))):
        item = dataset[i]
        print(f"--- Sample {i + 1} ---")
        print("PROMPT:", item.prompt[:250].replace("\n", "\\n"))
        print("CHOSEN:", item.chosen[:200].replace("\n", "\\n"))
        print("REJECTED:", item.rejected[:200].replace("\n", "\\n"))


def _build_response_mask(
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    batch, width = attention_mask.shape
    result = torch.zeros((batch, width), dtype=torch.bool)
    for i in range(batch):
        seq_len = int(attention_mask[i].sum().item())
        if seq_len <= 0:
            continue
        start = width - seq_len if padding_side == "left" else 0
        prompt_len = min(int(prompt_lens[i].item()), seq_len)
        response_start = start + prompt_len
        response_end = start + seq_len
        if response_start < response_end:
            result[i, response_start:response_end] = True
    return result


class SFTCollator:
    """Builds labels where prompt tokens are masked out (label=-100)."""

    def __init__(self, tokenizer: Any, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[PreferenceTriple]) -> dict[str, Any]:
        prompts = [x.prompt for x in batch]
        responses = [x.chosen for x in batch]
        full_texts = [p + r for p, r in zip(prompts, responses, strict=True)]

        enc = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        prompt_lens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"].shape[1]
        # Per-example prompt lengths can vary; recompute without batching for exactness.
        per_prompt_lens = torch.tensor(
            [
                len(
                    self.tokenizer(
                        p,
                        max_length=self.max_length,
                        truncation=True,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for p in prompts
            ],
            dtype=torch.long,
        )

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        batch_size, width = labels.shape
        for i in range(batch_size):
            seq_len = int(enc["attention_mask"][i].sum().item())
            if seq_len <= 0:
                continue
            start = width - seq_len if self.tokenizer.padding_side == "left" else 0
            cutoff = start + min(int(per_prompt_lens[i].item()), seq_len)
            labels[i, :cutoff] = -100

        response_mask = _build_response_mask(enc["attention_mask"], per_prompt_lens, self.tokenizer.padding_side)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "response_mask": response_mask,
            "prompt": prompts,
            "chosen": responses,
            "full_text": full_texts,
            "prompt_lengths": per_prompt_lens,
            "max_prompt_length_in_batch": prompt_lens,
        }


class RewardModelCollator:
    def __init__(self, tokenizer: Any, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[PreferenceTriple]) -> dict[str, Any]:
        chosen_texts = [x.prompt + x.chosen for x in batch]
        rejected_texts = [x.prompt + x.rejected for x in batch]

        chosen_enc = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "prompt": [x.prompt for x in batch],
            "chosen": [x.chosen for x in batch],
            "rejected": [x.rejected for x in batch],
        }


class DPOCollator:
    def __init__(self, tokenizer: Any, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[PreferenceTriple]) -> dict[str, Any]:
        prompts = [x.prompt for x in batch]
        chosen_texts = [x.prompt + x.chosen for x in batch]
        rejected_texts = [x.prompt + x.rejected for x in batch]

        chosen_enc = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        prompt_lens = torch.tensor(
            [
                len(
                    self.tokenizer(
                        p,
                        max_length=self.max_length,
                        truncation=True,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for p in prompts
            ],
            dtype=torch.long,
        )
        chosen_response_mask = _build_response_mask(
            chosen_enc["attention_mask"],
            prompt_lens=prompt_lens,
            padding_side=self.tokenizer.padding_side,
        )
        rejected_response_mask = _build_response_mask(
            rejected_enc["attention_mask"],
            prompt_lens=prompt_lens,
            padding_side=self.tokenizer.padding_side,
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_response_mask": chosen_response_mask,
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_response_mask": rejected_response_mask,
            "prompt": prompts,
            "chosen": [x.chosen for x in batch],
            "rejected": [x.rejected for x in batch],
            "prompt_lengths": prompt_lens,
        }


def build_dataloader(
    dataset: Dataset[Any],
    collator: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def preview_parsing(examples: Iterable[dict[str, Any]], k: int = 3) -> None:
    for i, row in enumerate(examples):
        if i >= k:
            break
        triple = parse_preference_example(row)
        print(f"Sample {i + 1}")
        print("Prompt:", triple.prompt[-250:].replace("\n", "\\n"))
        print("Chosen:", triple.chosen[:160].replace("\n", "\\n"))
        print("Rejected:", triple.rejected[:160].replace("\n", "\\n"))
