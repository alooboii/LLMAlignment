from __future__ import annotations

import textwrap


def main() -> None:
    message = textwrap.dedent(
        """
        LLM Alignment assignment runner.

        Use one of:
          python task_c0.py    # Task C0
          python train_rm.py   # Task C1
          python train_sft.py  # Task C2
          python train_rl.py --method ppo|dpo|grpo|rlvr  # Tasks C3/C4/C5/C6
          python eval.py ...   # Task C8
        """
    ).strip()
    print(message)


if __name__ == "__main__":
    main()
