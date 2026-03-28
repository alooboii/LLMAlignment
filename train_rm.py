from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import RewardModelCollator, build_dataloader, build_hh_datasets
from model.loading import count_parameters, describe_cuda_memory, load_reward_model_and_tokenizer
from model.lora import apply_lora, build_lora_config
from model.reward import pairwise_reward_loss, reward_scores
from utils import move_batch_to_device, set_seed


def evaluate_reward_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total, correct = 0, 0
    chosen_scores_all: list[float] = []
    rejected_scores_all: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            chosen_scores = reward_scores(
                model=model,
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            rejected_scores = reward_scores(
                model=model,
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.numel()
            chosen_scores_all.extend(chosen_scores.detach().float().cpu().tolist())
            rejected_scores_all.extend(rejected_scores.detach().float().cpu().tolist())

    pref_acc = correct / max(total, 1)
    chosen_t = torch.tensor(chosen_scores_all, dtype=torch.float32)
    rejected_t = torch.tensor(rejected_scores_all, dtype=torch.float32)
    return {
        "preference_accuracy": pref_acc,
        "chosen_mean": chosen_t.mean().item() if chosen_t.numel() else 0.0,
        "chosen_std": chosen_t.std(unbiased=False).item() if chosen_t.numel() else 0.0,
        "rejected_mean": rejected_t.mean().item() if rejected_t.numel() else 0.0,
        "rejected_std": rejected_t.std(unbiased=False).item() if rejected_t.numel() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task C1: Reward model training")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="checkpoints/reward_model")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-bits", type=int, choices=[4, 8], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
    print(f"Loaded HH-RLHF harmless: train={len(train_ds)} test={len(test_ds)}")

    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        model_name=args.model_name,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    )
    rm_model.to(device)
    rm_tokenizer.padding_side = "right"
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    if rm_model.config.pad_token_id is None:
        rm_model.config.pad_token_id = rm_tokenizer.pad_token_id

    if args.use_lora:
        lora_cfg = build_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=("q_proj", "v_proj"),
            task_type="seq_cls",
        )
        rm_model = apply_lora(rm_model, lora_cfg)
        rm_model.to(device)

    stats = count_parameters(rm_model)
    print(
        f"Reward model params: total={stats.total_params:,} trainable={stats.trainable_params:,} "
        f"({stats.trainable_ratio:.3%})"
    )
    print(describe_cuda_memory(prefix="RM memory: "))

    collator = RewardModelCollator(tokenizer=rm_tokenizer, max_length=args.max_length)
    train_loader = build_dataloader(train_ds, collator, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_ds, collator, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW((p for p in rm_model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    rm_model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"RM epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)
            out = pairwise_reward_loss(rm_model, batch, reg_lambda=args.reg_lambda)
            (out.loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix(
                loss=f"{out.loss.item():.4f}",
                pref_acc=f"{out.preference_accuracy.item():.3f}",
                step=global_step,
            )

    metrics = evaluate_reward_model(rm_model, test_loader, device=device)
    print(
        "Test preference_acc={preference_accuracy:.4f} chosen_mean={chosen_mean:.3f} "
        "chosen_std={chosen_std:.3f} rejected_mean={rejected_mean:.3f} "
        "rejected_std={rejected_std:.3f}".format(**metrics)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rm_model.save_pretrained(output_dir)
    rm_tokenizer.save_pretrained(output_dir)
    print(f"Saved reward model to {output_dir}")


if __name__ == "__main__":
    main()

