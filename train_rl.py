from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from tqdm import tqdm

from alignment.common import build_next_token_mask, forward_token_log_probs, masked_mean
from alignment.dpo import dpo_forward_pass
from alignment.grpo import (
    broadcast_group_advantages_to_tokens,
    degenerate_group_fraction,
    grpo_policy_loss,
    group_relative_advantages,
    standardize_token_advantages,
)
from alignment.ppo import (
    build_token_rewards,
    clipping_sanity_test,
    compute_gae,
    compute_token_log_probs_and_entropy,
    gae_unit_test,
    ppo_losses,
    ratio_sanity_test,
    standardize_advantages,
)
from data.gsm8k import extract_gold_answer, extract_numeric_answer, format_gsm8k_prompt, load_gsm8k, verifiable_reward
from data.hh_rlhf import DPOCollator, PreferenceTriple, build_dataloader, build_hh_datasets
from model.loading import (
    count_parameters,
    describe_cuda_memory,
    load_policy_or_adapter,
    load_reward_model_and_tokenizer,
)
from model.lora import apply_lora, build_lora_config, freeze_model
from model.reward import reward_scores
from model.value import ValueModel
from utils import move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tasks C3/C4/C5/C6: RL training")
    parser.add_argument("--method", type=str, choices=["ppo", "dpo", "grpo", "rlvr"], required=True)
    parser.add_argument("--policy-init", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--policy-ref", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/aligned")

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DPO or prompts-per-step for PPO/GRPO.")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1, help="Used by DPO.")
    parser.add_argument("--update-steps", type=int, default=200, help="Used by PPO/GRPO.")
    parser.add_argument("--mini-epochs", type=int, default=4, help="PPO/GRPO inner epochs.")
    parser.add_argument("--group-size", type=int, default=4, help="K for GRPO/RLVR.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.1, help="KL coefficient or DPO beta.")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)

    parser.add_argument("--lr-policy", type=float, default=1e-5)
    parser.add_argument("--lr-value", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-bits", type=int, choices=[4, 8], default=None)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)

    parser.add_argument("--reward-model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--reward-model-path", type=str, default=None)
    parser.add_argument("--value-model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--rlvr-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def start_resource_tracking() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_resource_metrics(step_durations: list[float], total_seconds: float) -> dict[str, float]:
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    avg_step_sec = sum(step_durations) / max(len(step_durations), 1)
    return {
        "peak_vram_gb": peak_vram_gb,
        "avg_step_sec": avg_step_sec,
        "total_train_sec": total_seconds,
        "num_update_steps": float(len(step_durations)),
    }


def save_run_metrics(out_dir: Path, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def sample_prompts(
    dataset: Sequence[PreferenceTriple],
    batch_size: int,
) -> list[PreferenceTriple]:
    idxs = random.sample(range(len(dataset)), k=min(batch_size, len(dataset)))
    return [dataset[i] for i in idxs]


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    enc = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_width = input_ids.size(1)

    with torch.no_grad():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        generated = model.generate(**gen_kwargs)

    generated_len = max(generated.size(1) - input_width, 0)
    response_mask = torch.zeros_like(generated, dtype=torch.bool)
    if generated_len > 0:
        response_mask[:, -generated_len:] = True

    full_attention = torch.cat(
        [attention_mask, torch.ones((attention_mask.size(0), generated_len), device=device, dtype=attention_mask.dtype)],
        dim=1,
    )
    response_ids = generated[:, -generated_len:] if generated_len > 0 else generated[:, :0]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    full_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return {
        "generated_ids": generated,
        "full_attention_mask": full_attention,
        "response_mask": response_mask,
        "responses": responses,
        "full_text": full_text,
        "input_width": input_width,
    }


def score_with_reward_model(
    *,
    rm_model: torch.nn.Module,
    rm_tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    texts = [p + r for p, r in zip(prompts, responses, strict=True)]
    enc = rm_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return reward_scores(
            rm_model,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )


def evaluate_rm_winrate_and_kl(
    *,
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    baseline: torch.nn.Module,
    tokenizer: Any,
    rm_model: torch.nn.Module,
    rm_tokenizer: Any,
    eval_prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    aligned = generate_batch(
        policy,
        tokenizer,
        eval_prompts,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    baseline_gen = generate_batch(
        baseline,
        tokenizer,
        eval_prompts,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    aligned_scores = score_with_reward_model(
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        prompts=eval_prompts,
        responses=aligned["responses"],
        max_length=max_length,
        device=device,
    )
    baseline_scores = score_with_reward_model(
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        prompts=eval_prompts,
        responses=baseline_gen["responses"],
        max_length=max_length,
        device=device,
    )
    win_rate = (aligned_scores > baseline_scores).float().mean().item()

    with torch.no_grad():
        aligned_lp = forward_token_log_probs(
            policy,
            input_ids=aligned["generated_ids"],
            attention_mask=aligned["full_attention_mask"],
        )
        ref_lp = forward_token_log_probs(
            reference,
            input_ids=aligned["generated_ids"],
            attention_mask=aligned["full_attention_mask"],
        )
    token_mask = build_next_token_mask(aligned["response_mask"])
    kl_mc = masked_mean(aligned_lp - ref_lp, token_mask).item()
    return {
        "win_rate_vs_sft": win_rate,
        "rm_score_mean": aligned_scores.mean().item(),
        "kl_mc": kl_mc,
    }


def evaluate_gsm8k_pass1(
    *,
    policy: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    n_eval: int,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    rows = list(rows[:n_eval])
    prompts = [format_gsm8k_prompt(r["question"]) for r in rows]
    gold = [extract_gold_answer(r["answer"]) for r in rows]
    generated = generate_batch(
        policy,
        tokenizer,
        prompts,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    rewards = [verifiable_reward(pred, g) for pred, g in zip(generated["responses"], gold, strict=True)]
    compliance = [1.0 if extract_numeric_answer(text) is not None else 0.0 for text in generated["responses"]]
    return {
        "pass_at_1": float(sum(rewards) / max(len(rewards), 1)),
        "format_compliance": float(sum(compliance) / max(len(compliance), 1)),
    }


def load_policy_and_reference(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, Any, torch.nn.Module]:
    policy, tokenizer = load_policy_or_adapter(
        args.policy_init,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
        is_trainable_adapter=True,
        enable_gradient_checkpointing=True,
    )
    policy.to(device)

    if args.use_lora and not (Path(args.policy_init) / "adapter_config.json").exists():
        lora_cfg = build_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=("q_proj", "v_proj"),
            task_type="causal_lm",
        )
        policy = apply_lora(policy, lora_cfg)
        policy.to(device)

    ref_source = args.policy_ref or args.policy_init
    reference, _ = load_policy_or_adapter(
        ref_source,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
        is_trainable_adapter=False,
        enable_gradient_checkpointing=False,
    )
    reference.to(device)
    reference = freeze_model(reference)
    return policy, tokenizer, reference


def run_dpo(args: argparse.Namespace, device: torch.device) -> None:
    train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
    policy, tokenizer, reference = load_policy_and_reference(args, device)
    baseline = reference

    rm_model = None
    rm_tokenizer = None
    if args.reward_model_path or args.reward_model_name:
        rm_source = args.reward_model_path or args.reward_model_name
        rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
            model_name=rm_source,
            dtype=args.dtype,
            load_in_bits=args.load_in_bits,
            device_map=None,
        )
        rm_model = freeze_model(rm_model.to(device))

    collator = DPOCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = build_dataloader(train_ds, collator, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_ds, collator, batch_size=args.batch_size, shuffle=False)

    # DPO sanity check: before updates, delta_theta ~= delta_ref so implicit margin near 0.
    sanity_batch = next(iter(test_loader))
    sanity_batch = move_batch_to_device(sanity_batch, device)
    with torch.no_grad():
        sanity_out = dpo_forward_pass(policy, reference, batch=sanity_batch, beta=args.beta)
    print(
        f"DPO init sanity: implicit_margin={sanity_out.implicit_margin.item():.4f} "
        f"pref_acc={sanity_out.preference_accuracy.item():.3f} (expected around 0.5)"
    )

    optimizer = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr_policy, weight_decay=args.weight_decay)
    policy.train()
    step = 0
    step_durations: list[float] = []
    start_resource_tracking()
    total_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"DPO epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar, start=1):
            iter_start = time.perf_counter()
            batch = move_batch_to_device(batch, device)
            out = dpo_forward_pass(policy, reference, batch=batch, beta=args.beta)
            (out.loss / args.grad_accum).backward()
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                step_durations.append(time.perf_counter() - iter_start)

                if step % args.eval_every == 0:
                    pref = []
                    with torch.no_grad():
                        for j, tb in enumerate(test_loader):
                            if j >= max(args.eval_size // args.batch_size, 1):
                                break
                            tb = move_batch_to_device(tb, device)
                            tout = dpo_forward_pass(policy, reference, batch=tb, beta=args.beta)
                            pref.append(tout.preference_accuracy.item())
                    msg = f"[eval] step={step} pref_acc={sum(pref)/max(len(pref),1):.3f}"
                    if rm_model is not None and rm_tokenizer is not None:
                        eval_prompts = [test_ds[k].prompt for k in range(min(args.eval_size, len(test_ds)))]
                        m = evaluate_rm_winrate_and_kl(
                            policy=policy,
                            reference=reference,
                            baseline=baseline,
                            tokenizer=tokenizer,
                            rm_model=rm_model,
                            rm_tokenizer=rm_tokenizer,
                            eval_prompts=eval_prompts,
                            max_length=args.max_length,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                        )
                        msg += (
                            f" win_rate={m['win_rate_vs_sft']:.3f}"
                            f" rm_mean={m['rm_score_mean']:.3f}"
                            f" kl={m['kl_mc']:.4f}"
                        )
                    print(msg)

            pbar.set_postfix(loss=f"{out.loss.item():.4f}", z=f"{out.implicit_margin.item():.3f}", step=step)

    save_path = Path(args.output_dir) / "dpo"
    save_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    run_metrics = collect_resource_metrics(step_durations, total_seconds=time.perf_counter() - total_start)
    run_metrics["method"] = "dpo"
    save_run_metrics(save_path, run_metrics)
    print(f"Saved DPO policy to {save_path}")


def run_ppo(args: argparse.Namespace, device: torch.device) -> None:
    train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
    policy, tokenizer, reference = load_policy_and_reference(args, device)
    baseline = reference

    rm_source = args.reward_model_path or args.reward_model_name
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        model_name=rm_source,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    )
    rm_model = freeze_model(rm_model.to(device))

    value_model = ValueModel.from_pretrained_backbone(
        args.value_model_name,
        dtype=args.dtype,
        load_in_bits=args.load_in_bits,
        device_map=None,
    ).to(device)
    if args.use_lora:
        lora_cfg = build_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=("q_proj", "v_proj"),
            task_type="feature_extraction",
        )
        value_model.backbone = apply_lora(value_model.backbone, lora_cfg)  # type: ignore[assignment]
        value_model.to(device)

    pol_opt = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr_policy, weight_decay=args.weight_decay)
    val_opt = AdamW((p for p in value_model.parameters() if p.requires_grad), lr=args.lr_value, weight_decay=args.weight_decay)
    step_durations: list[float] = []
    start_resource_tracking()
    total_start = time.perf_counter()

    print("PPO sanity checks:")
    print("  GAE unit test:", gae_unit_test())
    print("  Clipping test:", clipping_sanity_test())

    prompt_pool = [x.prompt for x in train_ds]
    eval_prompts = [test_ds[i].prompt for i in range(min(args.eval_size, len(test_ds)))]

    for step in range(1, args.update_steps + 1):
        step_start = time.perf_counter()
        prompts = random.sample(prompt_pool, k=min(args.batch_size, len(prompt_pool)))
        rollout = generate_batch(
            policy,
            tokenizer,
            prompts,
            device=device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated_ids = rollout["generated_ids"]
        full_attention = rollout["full_attention_mask"]
        response_mask = rollout["response_mask"]
        token_mask = build_next_token_mask(response_mask)

        with torch.no_grad():
            old_log_probs = forward_token_log_probs(policy, input_ids=generated_ids, attention_mask=full_attention)
            ref_log_probs = forward_token_log_probs(reference, input_ids=generated_ids, attention_mask=full_attention)
            old_values = value_model(generated_ids, full_attention)[:, :-1]
            task_rewards = score_with_reward_model(
                rm_model=rm_model,
                rm_tokenizer=rm_tokenizer,
                prompts=prompts,
                responses=rollout["responses"],
                max_length=args.max_length,
                device=device,
            )

        token_rewards = build_token_rewards(
            task_rewards=task_rewards,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            response_token_mask=token_mask,
            kl_coef=args.beta,
        )
        advantages, returns = compute_gae(
            rewards=token_rewards,
            values=old_values,
            response_token_mask=token_mask,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = standardize_advantages(advantages, token_mask)
        with torch.no_grad():
            start_epoch_log_probs = forward_token_log_probs(policy, input_ids=generated_ids, attention_mask=full_attention)
        print_ratio = ratio_sanity_test(old_log_probs, start_epoch_log_probs)

        for _ in range(args.mini_epochs):
            new_log_probs, token_entropy = compute_token_log_probs_and_entropy(
                policy,
                input_ids=generated_ids,
                attention_mask=full_attention,
            )
            new_values = value_model(generated_ids, full_attention)[:, :-1]
            losses = ppo_losses(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                returns=returns,
                new_values=new_values,
                response_token_mask=token_mask,
                clip_epsilon=args.clip_eps,
                value_coef=args.vf_coef,
                entropy_coef=args.entropy_coef,
                token_entropies=token_entropy,
            )
            pol_opt.zero_grad(set_to_none=True)
            val_opt.zero_grad(set_to_none=True)
            losses.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            pol_opt.step()
            val_opt.step()

        msg = (
            f"[ppo] step={step} reward={task_rewards.mean().item():.3f} "
            f"kl={losses.approx_kl_ref.item():.4f} "
            f"policy={losses.policy_loss.item():.4f} value={losses.value_loss.item():.4f} "
            f"ratio_check={print_ratio}"
        )
        print(msg)

        if step % args.eval_every == 0:
            m = evaluate_rm_winrate_and_kl(
                policy=policy,
                reference=reference,
                baseline=baseline,
                tokenizer=tokenizer,
                rm_model=rm_model,
                rm_tokenizer=rm_tokenizer,
                eval_prompts=eval_prompts,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print(
                f"[eval] step={step} win_rate={m['win_rate_vs_sft']:.3f} "
                f"rm_mean={m['rm_score_mean']:.3f} kl={m['kl_mc']:.4f}"
            )
        step_durations.append(time.perf_counter() - step_start)

    out_dir = Path(args.output_dir) / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    run_metrics = collect_resource_metrics(step_durations, total_seconds=time.perf_counter() - total_start)
    run_metrics["method"] = "ppo"
    save_run_metrics(out_dir, run_metrics)
    print(f"Saved PPO policy to {out_dir}")


def run_grpo_like(args: argparse.Namespace, device: torch.device, *, use_rlvr: bool) -> None:
    policy, tokenizer, reference = load_policy_and_reference(args, device)
    baseline = reference

    if use_rlvr:
        train_rows = list(load_gsm8k("train"))
        test_rows = list(load_gsm8k("test"))
        prompt_pool = [format_gsm8k_prompt(r["question"]) for r in train_rows]
        gold_pool = [extract_gold_answer(r["answer"]) for r in train_rows]
        eval_payload = test_rows
        max_new_tokens = args.rlvr_max_new_tokens
        rm_model = rm_tokenizer = None
    else:
        train_ds, test_ds = build_hh_datasets(train_limit=args.train_limit, test_limit=args.test_limit)
        prompt_pool = [x.prompt for x in train_ds]
        gold_pool = [None] * len(prompt_pool)
        eval_payload = [test_ds[i].prompt for i in range(min(args.eval_size, len(test_ds)))]
        max_new_tokens = args.max_new_tokens
        rm_source = args.reward_model_path or args.reward_model_name
        rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
            model_name=rm_source,
            dtype=args.dtype,
            load_in_bits=args.load_in_bits,
            device_map=None,
        )
        rm_model = freeze_model(rm_model.to(device))

    pol_opt = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr_policy, weight_decay=args.weight_decay)
    mode_name = "rlvr" if use_rlvr else "grpo"
    step_durations: list[float] = []
    start_resource_tracking()
    total_start = time.perf_counter()

    for step in range(1, args.update_steps + 1):
        step_start = time.perf_counter()
        batch_size = min(args.batch_size, len(prompt_pool))
        idx = random.sample(range(len(prompt_pool)), k=batch_size)
        base_prompts = [prompt_pool[i] for i in idx]
        base_gold = [gold_pool[i] for i in idx]

        # Expand each prompt K times.
        expanded_prompts: list[str] = []
        expanded_gold: list[str | None] = []
        for p, g in zip(base_prompts, base_gold, strict=True):
            expanded_prompts.extend([p] * args.group_size)
            expanded_gold.extend([g] * args.group_size)

        rollout = generate_batch(
            policy,
            tokenizer,
            expanded_prompts,
            device=device,
            max_length=min(args.max_length, 200 if use_rlvr else args.max_length),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated_ids = rollout["generated_ids"]
        full_attention = rollout["full_attention_mask"]
        response_mask = rollout["response_mask"]
        token_mask = build_next_token_mask(response_mask)

        with torch.no_grad():
            old_log_probs = forward_token_log_probs(policy, input_ids=generated_ids, attention_mask=full_attention)
            ref_log_probs = forward_token_log_probs(reference, input_ids=generated_ids, attention_mask=full_attention)

        if use_rlvr:
            rewards = torch.tensor(
                [verifiable_reward(pred, gold) for pred, gold in zip(rollout["responses"], expanded_gold, strict=True)],
                dtype=torch.float32,
                device=device,
            )
        else:
            rewards = score_with_reward_model(
                rm_model=rm_model,  # type: ignore[arg-type]
                rm_tokenizer=rm_tokenizer,  # type: ignore[arg-type]
                prompts=expanded_prompts,
                responses=rollout["responses"],
                max_length=args.max_length,
                device=device,
            )

        group_rewards = rewards.view(batch_size, args.group_size)
        group_adv = group_relative_advantages(group_rewards)
        token_adv = broadcast_group_advantages_to_tokens(
            group_advantages=group_adv,
            response_token_mask=token_mask,
        )
        token_adv = standardize_token_advantages(token_adv, token_mask)
        deg_frac = degenerate_group_fraction(group_rewards)

        for _ in range(args.mini_epochs):
            new_log_probs = forward_token_log_probs(policy, input_ids=generated_ids, attention_mask=full_attention)
            loss_out = grpo_policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                token_advantages=token_adv,
                response_token_mask=token_mask,
                clip_epsilon=args.clip_eps,
                group_rewards=group_rewards,
            )
            # KL regularization comes from reward shaping term.
            kl_bonus = masked_mean(new_log_probs - ref_log_probs, token_mask)
            total_loss = loss_out.total_loss + args.beta * kl_bonus

            pol_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            pol_opt.step()

        mean_length = float(response_mask.float().sum(dim=1).mean().item())
        print(
            f"[{mode_name}] step={step} mean_reward={rewards.mean().item():.3f} "
            f"degenerate={deg_frac:.3f} kl={loss_out.approx_kl_ref.item():.4f} "
            f"len={mean_length:.1f}"
        )

        if step % args.eval_every == 0:
            if use_rlvr:
                m = evaluate_gsm8k_pass1(
                    policy=policy,
                    tokenizer=tokenizer,
                    rows=eval_payload,  # type: ignore[arg-type]
                    n_eval=args.eval_size,
                    max_length=min(args.max_length, 200),
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                print(
                    f"[eval] step={step} pass@1={m['pass_at_1']:.3f} "
                    f"format={m['format_compliance']:.3f}"
                )
            else:
                m = evaluate_rm_winrate_and_kl(
                    policy=policy,
                    reference=reference,
                    baseline=baseline,
                    tokenizer=tokenizer,
                    rm_model=rm_model,  # type: ignore[arg-type]
                    rm_tokenizer=rm_tokenizer,  # type: ignore[arg-type]
                    eval_prompts=eval_payload,  # type: ignore[arg-type]
                    max_length=args.max_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                print(
                    f"[eval] step={step} win_rate={m['win_rate_vs_sft']:.3f} "
                    f"rm_mean={m['rm_score_mean']:.3f} kl={m['kl_mc']:.4f}"
                )
        step_durations.append(time.perf_counter() - step_start)

    out_dir = Path(args.output_dir) / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    run_metrics = collect_resource_metrics(step_durations, total_seconds=time.perf_counter() - total_start)
    run_metrics["method"] = mode_name
    save_run_metrics(out_dir, run_metrics)
    print(f"Saved {mode_name.upper()} policy to {out_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running method={args.method} on device={device}")
    print(describe_cuda_memory(prefix="Startup memory: "))

    start = time.perf_counter()
    if args.method == "dpo":
        run_dpo(args, device)
    elif args.method == "ppo":
        run_ppo(args, device)
    elif args.method == "grpo":
        run_grpo_like(args, device, use_rlvr=False)
    elif args.method == "rlvr":
        run_grpo_like(args, device, use_rlvr=True)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    elapsed = time.perf_counter() - start
    print(f"Completed {args.method} in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
