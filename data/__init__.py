"""Data loading and collation utilities for alignment experiments."""

from .gsm8k import (
    extract_gold_answer,
    extract_numeric_answer,
    format_gsm8k_prompt,
    load_gsm8k,
    verifiable_reward,
)
from .hh_rlhf import (
    DPOCollator,
    PreferenceTriple,
    RewardModelCollator,
    SFTCollator,
    build_hh_datasets,
    load_hh_harmless,
    print_sample_triples,
)

__all__ = [
    "DPOCollator",
    "PreferenceTriple",
    "RewardModelCollator",
    "SFTCollator",
    "build_hh_datasets",
    "extract_gold_answer",
    "extract_numeric_answer",
    "format_gsm8k_prompt",
    "load_gsm8k",
    "load_hh_harmless",
    "print_sample_triples",
    "verifiable_reward",
]
