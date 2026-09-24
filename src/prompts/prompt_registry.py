"""
Universal Deterministic Prompt Registry
======================================

Purpose
-------
Provides centralized, reproducible, and deterministic prompt templates for
all PromptEnergy-Bench task families:
  1. Mathematical Reasoning (e.g. GSM8K)
  2. Knowledge-Intensive QA (e.g. Natural Questions)
  3. Long-Context Question Answering
  4. Text Summarization (e.g. CNN/DailyMail)

Experimental Principles
-----------------------
1. ZERO_SHOT conditions must NOT contain training examples.
2. FEW_SHOT conditions explicitly contain fixed, isolated training split demonstrations.
3. CoT conditions explicitly request reasoning (short, standard, long).
4. Context is NEVER silently added unless the experiment explicitly requests it.
5. Prompt configurations are versioned and SHA-256 hashable for scientific reproducibility.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import hashlib
import json


# ============================================================
# Core Prompt Strategies
# ============================================================

class PromptStrategy(str, Enum):
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    FEW_SHOT_3 = "few_shot_3"
    ZERO_SHOT_COT = "zero_shot_cot"
    SHORT_COT = "short_cot"
    LONG_COT = "long_cot"
    RAG = "rag"


class TaskFamily(str, Enum):
    MATHEMATICS = "mathematics"
    KNOWLEDGE_QA = "knowledge_qa"
    LONG_CONTEXT_QA = "long_context_qa"
    SUMMARIZATION = "summarization"


# ============================================================
# GSM8K (Mathematics) Prompt Templates & Demonstrations
# ============================================================

FIXED_FEW_SHOT_EXAMPLES_GSM8K: List[Dict[str, str]] = [
    {
        "problem": (
            "Natalia sold clips to 48 of her friends in April, and then "
            "she sold half as many clips in May. How many clips did Natalia "
            "sell altogether in April and May?"
        ),
        "answer": "#### 72",
    },
    {
        "problem": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just "
            "did 50 minutes of babysitting. How much did she earn?"
        ),
        "answer": "#### 10",
    },
    {
        "problem": (
            "Betty is saving money for a new wallet which costs $100. "
            "Betty has only half of the money she needs. Her parents "
            "decided to give her $15 for that purpose, and her grandparents "
            "twice as much as her parents. How much more money does Betty "
            "need to buy the wallet?"
        ),
        "answer": "#### 5",
    },
]

# Alias for backward-compatibility
FIXED_FEW_SHOT_EXAMPLES = FIXED_FEW_SHOT_EXAMPLES_GSM8K


SYSTEM_GSM8K_ZERO_SHOT_DIRECT = (
    "You are a mathematical question-answering assistant.\n"
    "Solve the given GSM8K problem.\n"
    "Return only the final numerical answer.\n"
    "Do not provide explanations or intermediate calculations.\n"
    "Your response must end in the exact format:\n"
    "#### [number]"
)

SYSTEM_GSM8K_FEW_SHOT_3 = (
    "You are a mathematical question-answering assistant.\n"
    "Use the provided examples only as demonstrations of the expected "
    "answer format and problem-solving style.\n"
    "Solve the target problem independently.\n"
    "Return only the final numerical answer.\n"
    "Do not provide intermediate reasoning.\n"
    "Your response must end in the exact format:\n"
    "#### [number]"
)

SYSTEM_GSM8K_ZERO_SHOT_COT = (
    "You are a mathematical reasoning assistant.\n"
    "Solve the given GSM8K problem step by step.\n"
    "Show the necessary calculations and reasoning.\n"
    "After completing the reasoning, provide the final numerical answer.\n"
    "The final answer must use the exact format:\n"
    "#### [number]"
)

SYSTEM_GSM8K_SHORT_COT = (
    "You are a mathematical reasoning assistant.\n"
    "Solve the given GSM8K problem using at most 2 concise reasoning steps.\n"
    "Include only the calculations or deductions necessary to reach the "
    "answer.\n"
    "After the reasoning, provide the final numerical answer.\n"
    "The final answer must use the exact format:\n"
    "#### [number]"
)

SYSTEM_GSM8K_LONG_COT = (
    "You are a mathematical reasoning assistant.\n"
    "Solve the given GSM8K problem using detailed mathematical reasoning.\n"
    "Show the necessary calculations and intermediate deductions.\n"
    "Do not skip important reasoning steps.\n"
    "After completing the reasoning, provide the final numerical answer.\n"
    "The final answer must use the exact format:\n"
    "#### [number]"
)

SYSTEM_GSM8K_RAG = (
    "You are a mathematical question-answering assistant.\n"
    "Use the provided context to solve the given GSM8K problem.\n"
    "After completing any calculations, provide the final numerical answer.\n"
    "The final answer must use the exact format:\n"
    "#### [number]"
)


# ============================================================
# Knowledge QA (e.g. Natural Questions) System Prompts
# ============================================================

SYSTEM_QA_ZERO_SHOT_DIRECT = (
    "You are a factual question-answering assistant.\n"
    "Answer the following question directly and concisely.\n"
    "Provide only the factual answer without conversational filler."
)

SYSTEM_QA_ZERO_SHOT_COT = (
    "You are a reasoning question-answering assistant.\n"
    "Analyze the question step-by-step to arrive at the factual answer.\n"
    "Conclude with your final answer."
)

SYSTEM_QA_RAG = (
    "You are a knowledge-grounded question-answering assistant.\n"
    "Answer the target question using only the provided context.\n"
    "If the context does not contain the answer, state that clearly."
)


# ============================================================
# Summarization (e.g. CNN/DailyMail) System Prompts
# ============================================================

SYSTEM_SUMMARIZATION_DIRECT = (
    "You are an expert text summarization assistant.\n"
    "Write a concise summary highlighting the primary points of the given text."
)

SYSTEM_SUMMARIZATION_COT = (
    "You are an analytical text summarization assistant.\n"
    "First outline the key findings and narrative structure of the text,\n"
    "then provide a comprehensive summary."
)


# ============================================================
# User Prompt Builders
# ============================================================

def build_gsm8k_user_prompt(strategy: PromptStrategy, question: str) -> str:
    """Builds user prompt content for GSM8K problems."""
    if strategy == PromptStrategy.ZERO_SHOT_DIRECT:
        return f"Problem:\n{question}\n\nAnswer:"
    elif strategy == PromptStrategy.ZERO_SHOT_COT:
        return f"Problem:\n{question}\n\nLet's solve this step by step."
    elif strategy == PromptStrategy.SHORT_COT:
        return f"Problem:\n{question}\n\nReasoning:"
    elif strategy == PromptStrategy.LONG_COT:
        return f"Problem:\n{question}\n\nDetailed reasoning:"
    elif strategy == PromptStrategy.FEW_SHOT_3:
        parts = []
        for i, example in enumerate(FIXED_FEW_SHOT_EXAMPLES_GSM8K, start=1):
            parts.append(
                f"Example {i}\n"
                f"Problem:\n"
                f"{example['problem']}\n\n"
                f"Answer:\n"
                f"{example['answer']}\n"
            )
        parts.append(
            f"Target Problem:\n"
            f"{question}\n\n"
            f"Answer:"
        )
        return "\n".join(parts)
    else:
        return f"Problem:\n{question}\n\nAnswer:"


def build_qa_user_prompt(strategy: PromptStrategy, question: str) -> str:
    """Builds user prompt content for general QA."""
    if strategy in (PromptStrategy.ZERO_SHOT_COT, PromptStrategy.SHORT_COT, PromptStrategy.LONG_COT):
        return f"Question:\n{question}\n\nLet's think step by step."
    return f"Question:\n{question}\n\nAnswer:"


def build_summarization_user_prompt(strategy: PromptStrategy, text: str) -> str:
    """Builds user prompt content for summarization tasks."""
    return f"Document:\n{text}\n\nSummary:"


# ============================================================
# Prompt Registry Data Structure
# ============================================================

DATASET_PROMPT_REGISTRY: Dict[str, Dict[PromptStrategy, Dict[str, Any]]] = {
    "gsm8k": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_GSM8K_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.FEW_SHOT_3: {
            "system": SYSTEM_GSM8K_FEW_SHOT_3,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.FEW_SHOT_3, q),
            "uses_demonstrations": True,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_GSM8K_ZERO_SHOT_COT,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.ZERO_SHOT_COT, q),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.SHORT_COT: {
            "system": SYSTEM_GSM8K_SHORT_COT,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.SHORT_COT, q),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.LONG_COT: {
            "system": SYSTEM_GSM8K_LONG_COT,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.LONG_COT, q),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_GSM8K_RAG,
            "builder": lambda q: build_gsm8k_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
    },
    "natural_questions": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_QA_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_QA_ZERO_SHOT_COT,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.ZERO_SHOT_COT, q),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_QA_RAG,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.RAG, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        }
    },
    "cnn_dailymail": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_SUMMARIZATION_DIRECT,
            "builder": lambda t: build_summarization_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, t),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_SUMMARIZATION_COT,
            "builder": lambda t: build_summarization_user_prompt(PromptStrategy.ZERO_SHOT_COT, t),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        }
    },
    "contexteval": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_QA_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_QA_ZERO_SHOT_COT,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.ZERO_SHOT_COT, q),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": True,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_QA_RAG,
            "builder": lambda q: build_qa_user_prompt(PromptStrategy.RAG, q),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        }
    }
}

# Aliases
DATASET_PROMPT_REGISTRY["nq"] = DATASET_PROMPT_REGISTRY["natural_questions"]
DATASET_PROMPT_REGISTRY["cnn"] = DATASET_PROMPT_REGISTRY["cnn_dailymail"]
DATASET_PROMPT_REGISTRY["context_eval"] = DATASET_PROMPT_REGISTRY["contexteval"]

# Backward compatibility registry alias
PROMPT_REGISTRY = DATASET_PROMPT_REGISTRY["gsm8k"]


# ============================================================
# Universal Formatter
# ============================================================

def format_prompt(
    dataset: str,
    strategy: PromptStrategy,
    input_text: str,
    context: Optional[str] = None,
    allow_context: bool = False
) -> List[Dict[str, str]]:
    """Universal prompt formatter for all benchmark datasets and task families."""
    dataset_key = dataset.lower().strip()
    if dataset_key not in DATASET_PROMPT_REGISTRY:
        # Fallback to gsm8k if unknown
        dataset_key = "gsm8k"

    registry = DATASET_PROMPT_REGISTRY[dataset_key]
    if strategy not in registry:
        raise ValueError(f"Strategy '{strategy}' not supported for dataset '{dataset}'")

    config = registry[strategy]

    if context and not config["context_allowed"] and not allow_context:
        raise ValueError(
            f"Context supplied to '{strategy.value}', but strategy does not allow external context."
        )

    user_content = config["builder"](input_text)
    if context:
        user_content = f"Context:\n{context}\n\n{user_content}"

    return [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": user_content}
    ]


def format_gsm8k_prompt(
    strategy: PromptStrategy,
    question: str,
    context: Optional[str] = None,
    allow_context: bool = False
) -> List[Dict[str, str]]:
    """Specific helper for GSM8K dataset prompt formatting."""
    return format_prompt(
        dataset="gsm8k",
        strategy=strategy,
        input_text=question,
        context=context,
        allow_context=allow_context
    )


# ============================================================
# Metadata & Reproducibility Hashing
# ============================================================

def get_prompt_metadata(
    strategy: PromptStrategy,
    dataset: str = "gsm8k"
) -> Dict[str, Any]:
    """Return metadata describing the experimental prompt condition."""
    dataset_key = dataset.lower().strip()
    config = DATASET_PROMPT_REGISTRY.get(dataset_key, DATASET_PROMPT_REGISTRY["gsm8k"])[strategy]
    demos = len(FIXED_FEW_SHOT_EXAMPLES_GSM8K) if (dataset_key == "gsm8k" and config["uses_demonstrations"]) else 0

    return {
        "dataset": dataset_key,
        "strategy": strategy.value,
        "uses_demonstrations": config["uses_demonstrations"],
        "reasoning": config["reasoning"],
        "context_allowed": config["context_allowed"],
        "num_demonstrations": demos
    }


def get_prompt_hash(
    prompt_version: str = "promptenergy_v2.0",
    dataset: str = "gsm8k"
) -> str:
    """Compute a deterministic SHA-256 hash of the prompt registry."""
    dataset_key = dataset.lower().strip()
    registry = DATASET_PROMPT_REGISTRY.get(dataset_key, DATASET_PROMPT_REGISTRY["gsm8k"])

    payload = {
        "version": prompt_version,
        "dataset": dataset_key,
        "registry": {
            strat.value: {
                "system": cfg["system"],
                "uses_demonstrations": cfg["uses_demonstrations"],
                "reasoning": cfg["reasoning"],
                "context_allowed": cfg["context_allowed"]
            }
            for strat, cfg in registry.items()
        }
    }
    if dataset_key == "gsm8k":
        payload["few_shot_examples"] = FIXED_FEW_SHOT_EXAMPLES_GSM8K

    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_prompt_integrity() -> None:
    """Validates prompt integrity across all datasets."""
    assert len(FIXED_FEW_SHOT_EXAMPLES_GSM8K) == 3, "GSM8K FEW_SHOT_3 must have 3 demonstrations."
    for ex in FIXED_FEW_SHOT_EXAMPLES_GSM8K:
        assert "problem" in ex and "answer" in ex and ex["answer"].startswith("#### ")

    # GSM8K zero-shot validation
    gsm_reg = DATASET_PROMPT_REGISTRY["gsm8k"]
    assert gsm_reg[PromptStrategy.ZERO_SHOT_DIRECT]["uses_demonstrations"] is False
    assert gsm_reg[PromptStrategy.ZERO_SHOT_COT]["uses_demonstrations"] is False
    assert gsm_reg[PromptStrategy.FEW_SHOT_3]["uses_demonstrations"] is True
