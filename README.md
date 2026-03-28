## DVLM PA2 Coding Implementation

This repository now includes end-to-end code for the coding tasks in `DVLM_PA2.pdf`:

- `Task C0` data parsing, dataloaders, model loading, LoRA setup
- `Task C1` reward model training (`train_rm.py`)
- `Task C2` SFT warm-up (`train_sft.py`)
- `Task C3` PPO (`train_rl.py --method ppo`)
- `Task C4` DPO (`train_rl.py --method dpo`)
- `Task C5` GRPO (`train_rl.py --method grpo`)
- `Task C6` RLVR on GSM8K (`train_rl.py --method rlvr`)
- `Task C8` evaluation script (`eval.py`)

## Structure

- `data/` HH-RLHF parsing + collators, GSM8K loader + verifiable reward parsing
- `model/` policy/reward/value loading, LoRA helpers, reward/value heads
- `alignment/` PPO/DPO/GRPO/RLVR losses and utilities
- `train_rm.py` reward model loop
- `train_sft.py` SFT loop with prompt masking
- `train_rl.py` PPO/DPO/GRPO/RLVR training loops
- `eval.py` win-rate, KL, sample table, optional GSM8K pass@1
- `task_c0.py` C0 setup sanity script (prints parsed triples + memory/params)

## Quick Start

```bash
python task_c0.py
python train_rm.py --use-lora
python train_sft.py --use-lora
python train_rl.py --method ppo --policy-init checkpoints/sft/policy_init --policy-ref checkpoints/sft/policy_ref
python train_rl.py --method dpo --policy-init checkpoints/sft/policy_init --policy-ref checkpoints/sft/policy_ref
python train_rl.py --method grpo --policy-init checkpoints/sft/policy_init --policy-ref checkpoints/sft/policy_ref
python train_rl.py --method rlvr --policy-init checkpoints/sft/policy_init --policy-ref checkpoints/sft/policy_ref --beta 0.05 --update-steps 300
python eval.py --sft-path checkpoints/sft/policy_ref --candidates ppo=checkpoints/aligned/ppo dpo=checkpoints/aligned/dpo grpo=checkpoints/aligned/grpo rlvr=checkpoints/aligned/rlvr --eval-gsm8k
```

## Notes

- High-level trainers (TRL/OpenRLHF/etc.) are not used.
- PPO/GRPO cache old-policy logprobs from rollout.
- DPO uses response-token-only sequence log-prob sums.
- RLVR uses binary verifiable rewards from extracted GSM8K numeric answers.
