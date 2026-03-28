from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import SFTCollator, build_dataloader, build_hh_datasets
from model.loading import count_parameters, describe_cuda_memory, load_policy_and_tokenizer
from model.lora import apply_lora, build_lora_config
from utils import move_batch_to_device, set_seed


def evaluate_perplexity(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )
            losses.append(out.loss.item())
    model.train()
    if not losses:
        return float("inf")
    mean_loss = sum(losses) / len(losses)
    return float(math.exp(min(mean_loss, 30)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task C2: SFT warm-up")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
    print(f"Loaded HH-RLHF harmless: train={len(train_ds)} test={len(test_ds)}")

    policy, tokenizer = load_policy_and_tokenizer(
        model_name=args.model_name,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
        enable_gradient_checkpointing=True,
    )
    policy.to(device)

    if args.use_lora:
        lora_cfg = build_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=("q_proj", "v_proj"),
            task_type="causal_lm",
        )
        policy = apply_lora(policy, lora_cfg)
        policy.to(device)

    stats = count_parameters(policy)
    print(
        f"Policy params: total={stats.total_params:,} trainable={stats.trainable_params:,} "
        f"({stats.trainable_ratio:.3%})"
    )
    print(describe_cuda_memory(prefix="Policy memory: "))

    collator = SFTCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = build_dataloader(train_ds, collator, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_ds, collator, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    policy.train()

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"SFT epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)
            out = policy(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )
            (out.loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.eval_every == 0:
                    ppl = evaluate_perplexity(policy, test_loader, device=device)
                    print(f"[eval] step={global_step} ppl={ppl:.3f}")

            pbar.set_postfix(loss=f"{out.loss.item():.4f}", step=global_step)

    # Greedy sanity samples.
    policy.eval()
    sample_prompts = [test_ds[i].prompt for i in range(min(5, len(test_ds)))]
    encoded = tokenizer(sample_prompts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        generated = policy.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    for i, text in enumerate(decoded, start=1):
        print(f"\n[SAMPLE {i}]\n{text[:700]}")

    output_root = Path(args.output_dir)
    ref_dir = output_root / "policy_ref"
    init_dir = output_root / "policy_init"
    ref_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(ref_dir)
    tokenizer.save_pretrained(ref_dir)
    policy.save_pretrained(init_dir)
    tokenizer.save_pretrained(init_dir)
    print(f"Saved SFT reference to {ref_dir}")
    print(f"Saved SFT trainable init to {init_dir}")


if __name__ == "__main__":
    main()

