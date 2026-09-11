"""Prompt registry and standardized templates for PromptEnergy-Bench."""
from src.prompts.gsm8k_prompts import (
    PromptStrategy,
    format_gsm8k_prompt,
    get_prompt_hash,
    FIXED_FEW_SHOT_EXAMPLES
)

__all__ = [
    "PromptStrategy",
    "format_gsm8k_prompt",
    "get_prompt_hash",
    "FIXED_FEW_SHOT_EXAMPLES"
]
