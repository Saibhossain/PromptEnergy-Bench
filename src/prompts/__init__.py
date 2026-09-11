"""Prompt registry and standardized templates for PromptEnergy-Bench across all datasets."""
from src.prompts.prompt_registry import (
    PromptStrategy,
    TaskFamily,
    format_prompt,
    format_gsm8k_prompt,
    get_prompt_metadata,
    get_prompt_hash,
    validate_prompt_integrity,
    FIXED_FEW_SHOT_EXAMPLES_GSM8K,
    FIXED_FEW_SHOT_EXAMPLES,
    DATASET_PROMPT_REGISTRY,
    PROMPT_REGISTRY
)

__all__ = [
    "PromptStrategy",
    "TaskFamily",
    "format_prompt",
    "format_gsm8k_prompt",
    "get_prompt_metadata",
    "get_prompt_hash",
    "validate_prompt_integrity",
    "FIXED_FEW_SHOT_EXAMPLES_GSM8K",
    "FIXED_FEW_SHOT_EXAMPLES",
    "DATASET_PROMPT_REGISTRY",
    "PROMPT_REGISTRY"
]
