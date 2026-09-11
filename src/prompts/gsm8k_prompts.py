"""
GSM8K Prompt Module (Compatibility Re-export)
============================================
This module re-exports GSM8K prompt definitions from the universal
`src.prompts.prompt_registry` for backward compatibility.
"""

from src.prompts.prompt_registry import (
    PromptStrategy,
    TaskFamily,
    FIXED_FEW_SHOT_EXAMPLES,
    FIXED_FEW_SHOT_EXAMPLES_GSM8K,
    SYSTEM_GSM8K_ZERO_SHOT_DIRECT as SYSTEM_ZERO_SHOT_DIRECT,
    SYSTEM_GSM8K_FEW_SHOT_3 as SYSTEM_FEW_SHOT_3,
    SYSTEM_GSM8K_ZERO_SHOT_COT as SYSTEM_ZERO_SHOT_COT,
    SYSTEM_GSM8K_SHORT_COT as SYSTEM_SHORT_COT,
    SYSTEM_GSM8K_LONG_COT as SYSTEM_LONG_COT,
    build_gsm8k_user_prompt,
    PROMPT_REGISTRY,
    DATASET_PROMPT_REGISTRY,
    format_prompt,
    format_gsm8k_prompt,
    get_prompt_metadata,
    get_prompt_hash,
    validate_prompt_integrity,
)

__all__ = [
    "PromptStrategy",
    "TaskFamily",
    "FIXED_FEW_SHOT_EXAMPLES",
    "FIXED_FEW_SHOT_EXAMPLES_GSM8K",
    "SYSTEM_ZERO_SHOT_DIRECT",
    "SYSTEM_FEW_SHOT_3",
    "SYSTEM_ZERO_SHOT_COT",
    "SYSTEM_SHORT_COT",
    "SYSTEM_LONG_COT",
    "build_gsm8k_user_prompt",
    "PROMPT_REGISTRY",
    "DATASET_PROMPT_REGISTRY",
    "format_prompt",
    "format_gsm8k_prompt",
    "get_prompt_metadata",
    "get_prompt_hash",
    "validate_prompt_integrity",
]
