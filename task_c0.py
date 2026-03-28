from __future__ import annotations

import argparse

import torch

from data.hh_rlhf import build_hh_datasets, print_sample_triples
from model.loading import (
    count_parameters,
    describe_cuda_memory,
    load_backbone_model,
    load_policy_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from model.lora import apply_lora, build_lora_config, clone_and_freeze_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task C0 sanity: data + model loading + LoRA")
    parser.add_argument("--policy-model", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--backbone-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
    print(f"HH harmless loaded: train={len(train_ds)} test={len(test_ds)}")
    print_sample_triples(train_ds, k=3)

    policy, tokenizer = load_policy_and_tokenizer(
        model_name=args.policy_model,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
        enable_gradient_checkpointing=True,
    )
    policy.to(device)
    print(
        f"Policy tokenization: pad_token={tokenizer.pad_token!r} "
        f"pad_side={tokenizer.padding_side!r}"
    )
    stats = count_parameters(policy)
    print(
        f"Policy params: total={stats.total_params:,} trainable={stats.trainable_params:,} "
        f"ratio={stats.trainable_ratio:.3%}"
    )
    print(describe_cuda_memory(prefix="After policy load: "))

    lora_cfg = build_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=("q_proj", "v_proj"),
        task_type="causal_lm",
    )
    policy_lora = apply_lora(policy, lora_cfg)
    stats_lora = count_parameters(policy_lora)
    print(
        f"Policy+LoRA trainable params: {stats_lora.trainable_params:,} "
        f"({stats_lora.trainable_ratio:.3%} of total)"
    )

    ref_copy = clone_and_freeze_model(policy_lora)
    ref_stats = count_parameters(ref_copy)
    print(f"Reference copy trainable params: {ref_stats.trainable_params:,} (expected 0)")

    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        model_name=args.backbone_model,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    )
    rm_model.to(device)
    print(
        f"RM tokenizer: pad_token={rm_tokenizer.pad_token!r} "
        f"pad_side={rm_tokenizer.padding_side!r}"
    )
    rm_stats = count_parameters(rm_model)
    print(
        f"RM params: total={rm_stats.total_params:,} trainable={rm_stats.trainable_params:,} "
        f"ratio={rm_stats.trainable_ratio:.3%}"
    )
    print(describe_cuda_memory(prefix="After RM load: "))

    value_backbone = load_backbone_model(
        model_name=args.backbone_model,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    )
    value_backbone.to(device)
    value_stats = count_parameters(value_backbone)
    print(
        f"Value backbone params: total={value_stats.total_params:,} trainable={value_stats.trainable_params:,} "
        f"ratio={value_stats.trainable_ratio:.3%}"
    )
    print(describe_cuda_memory(prefix="After value backbone load: "))


if __name__ == "__main__":
    main()

