"""Centralized GSM8K Prompt Templates and Registry.

Enforces deterministic prompting across experiments, eliminating prompt dispersion.
"""

import hashlib
import json
from enum import Enum
from typing import Dict, List, Any


class PromptStrategy(str, Enum):
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    FEW_SHOT_3 = "few_shot_3"
    ZERO_SHOT_COT = "zero_shot_cot"
    SHORT_COT = "short_cot"
    LONG_COT = "long_cot"


# Three fixed deterministic GSM8K training examples (from train split records 0, 1, 2)
FIXED_FEW_SHOT_EXAMPLES = [
    {
        "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "#### 72"
    },
    {
        "problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "#### 10"
    },
    {
        "problem": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "#### 5"
    }
]


def _build_3shot_user_prompt(question: str) -> str:
    examples_str = ""
    for idx, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES, 1):
        examples_str += f"Example {idx}:\nProblem: {ex['problem']}\nAnswer: {ex['answer']}\n\n"

    return (
        f"{examples_str}"
        f"Target Problem:\n{question}\n\n"
        f"Return only:\n#### [number]"
    )


PROMPT_TEMPLATES = {
    PromptStrategy.ZERO_SHOT_DIRECT: {
        "system": (
            "You are a mathematical question-answering assistant.\n"
            "Solve the problem internally and return only the final numerical answer.\n"
            "Do not provide explanations, intermediate calculations, or additional text.\n"
            "Your response must follow exactly:\n"
            "#### [number]"
        ),
        "user_template": "Problem: {question}\n\nAnswer:"
    },
    PromptStrategy.FEW_SHOT_3: {
        "system": (
            "You are a mathematical question-answering assistant.\n"
            "Solve the target problem and return only the final numerical answer.\n"
            "Do not provide intermediate text. Return only:\n"
            "#### [number]"
        ),
        "user_builder": _build_3shot_user_prompt
    },
    PromptStrategy.ZERO_SHOT_COT: {
        "system": (
            "You are a mathematical reasoning assistant.\n"
            "Solve the problem step by step.\n"
            "Show necessary calculations and reasoning.\n"
            "End with the exact format:\n"
            "#### [number]"
        ),
        "user_template": "Problem: {question}\n\nLet's think step by step."
    },
    PromptStrategy.SHORT_COT: {
        "system": (
            "You are a mathematical reasoning assistant.\n"
            "Solve using at most 2 concise reasoning steps.\n"
            "Each step should contain only the necessary calculation or deduction.\n"
            "Do not include unnecessary explanations.\n"
            "End with:\n"
            "#### [number]"
        ),
        "user_template": "Problem: {question}\n\nReasoning:"
    },
    PromptStrategy.LONG_COT: {
        "system": (
            "You are a mathematical reasoning assistant.\n"
            "Solve using detailed mathematical reasoning.\n"
            "Show necessary calculations and intermediate deductions.\n"
            "Do not omit important reasoning steps.\n"
            "End with:\n"
            "#### [number]"
        ),
        "user_template": "Problem: {question}\n\nReasoning:"
    }
}


def format_gsm8k_prompt(
    strategy: PromptStrategy,
    question: str,
    context: str = ""
) -> List[Dict[str, str]]:
    """Formats standardized messages array for the model backend.
    
    If context is provided (e.g., in context scaling or RAG), it is prepended
    to the user prompt.
    """
    if strategy not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt strategy: {strategy}")

    tmpl = PROMPT_TEMPLATES[strategy]
    system_content = tmpl["system"]

    if "user_builder" in tmpl:
        user_content = tmpl["user_builder"](question)
    else:
        user_content = tmpl["user_template"].format(question=question)

    if context:
        user_content = f"{context}\n\n{user_content}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def get_prompt_hash(prompt_version: str = "v1.0") -> str:
    """Computes a deterministic hash of all prompt templates for reproducibility."""
    payload = {
        "version": prompt_version,
        "templates": {k.value: v["system"] for k, v in PROMPT_TEMPLATES.items()},
        "few_shot": FIXED_FEW_SHOT_EXAMPLES
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
