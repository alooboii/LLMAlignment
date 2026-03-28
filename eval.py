from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alignment.common import build_next_token_mask, forward_token_log_probs, masked_mean
from data.gsm8k import extract_gold_answer, format_gsm8k_prompt, load_gsm8k, verifiable_reward
from data.hh_rlhf import build_hh_datasets
from model.loading import load_policy_or_adapter, load_reward_model_and_tokenizer
from model.lora import freeze_model
from model.reward import reward_scores
from train_rl import generate_batch


@dataclass(slots=True)
class Candidate:
    name: str
    path: str


def parse_candidates(values: list[str]) -> list[Candidate]:
    out: list[Candidate] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Candidate must be name=path, got {item}")
        name, path = item.split("=", maxsplit=1)
        out.append(Candidate(name=name.strip(), path=path.strip()))
    return out


def score_batch_with_rm(
    rm_model: torch.nn.Module,
    rm_tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    texts = [p + r for p, r in zip(prompts, responses, strict=True)]
    enc = rm_tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return reward_scores(rm_model, enc["input_ids"], enc["attention_mask"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Task C8 evaluation script")
    parser.add_argument("--sft-path", type=str, required=True)
    parser.add_argument("--candidates", type=str, nargs="+", required=True, help="name=checkpoint_path")
    parser.add_argument("--reward-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--eval-gsm8k", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = parse_candidates(args.candidates)

    ref_model, tokenizer = load_policy_or_adapter(
        args.sft_path,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
        is_trainable_adapter=False,
        enable_gradient_checkpointing=False,
    )
    ref_model = freeze_model(ref_model.to(device))

    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        model_name=args.reward_model,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    )
    rm_model = freeze_model(rm_model.to(device))

    _, hh_test = build_hh_datasets(train_limit=0, test_limit=args.eval_size)
    prompts = [hh_test[i].prompt for i in range(min(args.eval_size, len(hh_test)))]

    sft_gen = generate_batch(
        ref_model,
        tokenizer,
        prompts,
        device=device,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    sft_scores = score_batch_with_rm(
        rm_model,
        rm_tokenizer,
        prompts,
        sft_gen["responses"],
        max_length=args.max_length,
        device=device,
    )

    print("=== RM Win-Rate vs SFT ===")
    resource_rows: list[tuple[str, float | None, float | None, float | None]] = []
    for cand in candidates:
        model, _ = load_policy_or_adapter(
            cand.path,
            dtype=args.dtype,
            load_in_bits=args.load_in_bits,
            device_map=None,
            is_trainable_adapter=False,
            enable_gradient_checkpointing=False,
        )
        model = freeze_model(model.to(device))
        generated = generate_batch(
            model,
            tokenizer,
            prompts,
            device=device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        scores = score_batch_with_rm(
            rm_model,
            rm_tokenizer,
            prompts,
            generated["responses"],
            max_length=args.max_length,
            device=device,
        )
        win_rate = (scores > sft_scores).float().mean().item()

        with torch.no_grad():
            lp_model = forward_token_log_probs(
                model,
                input_ids=generated["generated_ids"],
                attention_mask=generated["full_attention_mask"],
            )
            lp_ref = forward_token_log_probs(
                ref_model,
                input_ids=generated["generated_ids"],
                attention_mask=generated["full_attention_mask"],
            )
        token_mask = build_next_token_mask(generated["response_mask"])
        kl_mc = masked_mean(lp_model - lp_ref, token_mask).item()
        print(f"{cand.name:12s} win_rate={win_rate:.3f} rm_mean={scores.mean().item():.3f} kl={kl_mc:.4f}")
        metrics_path = Path(cand.path) / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            resource_rows.append(
                (
                    cand.name,
                    float(m.get("peak_vram_gb")) if m.get("peak_vram_gb") is not None else None,
                    float(m.get("avg_step_sec")) if m.get("avg_step_sec") is not None else None,
                    float(m.get("total_train_sec")) if m.get("total_train_sec") is not None else None,
                )
            )
        else:
            resource_rows.append((cand.name, None, None, None))

        print(f"\n--- Sample Table: {cand.name} ---")
        for i in range(min(args.sample_size, len(prompts))):
            print(f"[Prompt {i + 1}]")
            print(prompts[i][:220].replace("\n", " "))
            print("SFT:", sft_gen["responses"][i][:180].replace("\n", " "))
            print(f"{cand.name}:", generated["responses"][i][:180].replace("\n", " "))
            print(
                f"RM(SFT)={sft_scores[i].item():.3f} RM({cand.name})={scores[i].item():.3f}"
            )
            print()

    print("=== Resource Table ===")
    print("method        peak_vram_gb   sec_per_step   total_sec")
    for name, peak_vram, sec_per_step, total_sec in resource_rows:
        peak_s = f"{peak_vram:.2f}" if peak_vram is not None else "n/a"
        step_s = f"{sec_per_step:.3f}" if sec_per_step is not None else "n/a"
        total_s = f"{total_sec:.1f}" if total_sec is not None else "n/a"
        print(f"{name:12s} {peak_s:>12s} {step_s:>13s} {total_s:>10s}")

    if args.eval_gsm8k:
        gsm_test = list(load_gsm8k("test"))
        subset = gsm_test[: args.eval_size]
        gsm_prompts = [format_gsm8k_prompt(r["question"]) for r in subset]
        gold = [extract_gold_answer(r["answer"]) for r in subset]
        print("=== GSM8K pass@1 ===")
        for cand in candidates:
            model, _ = load_policy_or_adapter(
                cand.path,
                dtype=args.dtype,
                load_in_bits=args.load_in_bits,
                device_map=None,
                is_trainable_adapter=False,
                enable_gradient_checkpointing=False,
            )
            model = freeze_model(model.to(device))
            generated = generate_batch(
                model,
                tokenizer,
                gsm_prompts,
                device=device,
                max_length=min(args.max_length, 200),
                max_new_tokens=max(256, args.max_new_tokens),
                do_sample=False,
            )
            rewards = [
                verifiable_reward(pred, g)
                for pred, g in zip(generated["responses"], gold, strict=True)
            ]
            pass_at_1 = sum(rewards) / max(len(rewards), 1)
            print(f"{cand.name:12s} pass@1={pass_at_1:.3f}")


if __name__ == "__main__":
    main()
