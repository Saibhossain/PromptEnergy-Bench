"""
Universal Generalized Prompt Registry
====================================

Purpose
-------
Provides clean, publication-grade, task-agnostic prompt paradigms across all
PromptEnergy-Bench datasets and hardware benchmarks.

Prompting Paradigms (Meta-Strategies):
  1. ZERO_SHOT_DIRECT : Pure direct instruction. Direct question fed from dataset.
  2. FEW_SHOT_3       : In-context learning via 3 fixed, high-quality domain exemplars.
  3. ZERO_SHOT_COT    : Classic Kojima et al. "Let's think step by step" zero-shot reasoning.
  4. SHORT_COT        : Budget-constrained reasoning (concise 1-2 steps) for Green AI efficiency.
  5. LONG_COT         : Comprehensive, exhaustive step-by-step derivation.
  6. RAG              : Context-grounded generation utilizing retrieved or injected context.

Scientific Principles
---------------------
- Generalized and clean: No dataset-hardcoded jargon in system meta-prompts.
- Fully reproducible: Versioned and SHA-256 hashable.
- Zero data contamination: Demonstrations are strictly isolated from evaluation splits.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
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
# 1. Generalized Universal System Prompts (Publication Ready)
# ============================================================

SYSTEM_ZERO_SHOT_DIRECT = (
    "You are an expert AI assistant.\n"
    "Provide a direct, concise, and accurate answer to the user's question or task.\n"
    "Do not include unnecessary conversational filler, preambles, or explanations."
)

SYSTEM_FEW_SHOT_3 = (
    "You are an expert AI assistant.\n"
    "Carefully review the provided demonstration examples to understand the expected format and task requirements.\n"
    "Solve the target task directly and accurately, following the demonstrated style."
)

SYSTEM_ZERO_SHOT_COT = (
    "You are an expert AI reasoning assistant.\n"
    "Solve the given problem or task step by step, showing your logical deductions clearly.\n"
    "Conclude your reasoning with a clear and definite final answer."
)

SYSTEM_SHORT_COT = (
    "You are an expert AI reasoning assistant.\n"
    "Solve the given task using at most 1 to 2 concise reasoning steps.\n"
    "Keep your explanation brief and focused, then state the final answer."
)

SYSTEM_LONG_COT = (
    "You are an expert AI reasoning assistant.\n"
    "Provide a rigorous, detailed, and comprehensive step-by-step derivation.\n"
    "Explain all intermediate calculations, evidence, and logical deductions thoroughly before stating the final answer."
)

SYSTEM_RAG = (
    "You are a knowledge-grounded AI assistant.\n"
    "Answer the user's question or task strictly based on the provided reference context.\n"
    "If the context does not contain the answer, state that clearly. Be concise, factual, and accurate."
)


# ============================================================
# 2. Dataset-Specific Formatting Directives (Optional Suffixes)
# ============================================================

# For GSM8K Math: Standard delimiter used in literature (Cobbe et al.)
GSM8K_FORMAT_INSTRUCTION = "\nYour final numerical answer must end in the exact format: #### [number]"

# Specific system prompts with formatting rules preserved
SYSTEM_GSM8K_ZERO_SHOT_DIRECT = SYSTEM_ZERO_SHOT_DIRECT + GSM8K_FORMAT_INSTRUCTION
SYSTEM_GSM8K_FEW_SHOT_3 = SYSTEM_FEW_SHOT_3 + GSM8K_FORMAT_INSTRUCTION
SYSTEM_GSM8K_ZERO_SHOT_COT = SYSTEM_ZERO_SHOT_COT + GSM8K_FORMAT_INSTRUCTION
SYSTEM_GSM8K_SHORT_COT = SYSTEM_SHORT_COT + GSM8K_FORMAT_INSTRUCTION
SYSTEM_GSM8K_LONG_COT = SYSTEM_LONG_COT + GSM8K_FORMAT_INSTRUCTION
SYSTEM_GSM8K_RAG = SYSTEM_RAG + GSM8K_FORMAT_INSTRUCTION


# ============================================================
# 3. Fixed In-Context Learning Exemplars (3-Shot)
# ============================================================

FIXED_FEW_SHOT_EXAMPLES_GSM8K: List[Dict[str, str]] = [
    {
        "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "#### 72",
    },
    {
        "problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "#### 10",
    },
    {
        "problem": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "#### 5",
    },
]

# Alias for backward compatibility
FIXED_FEW_SHOT_EXAMPLES = FIXED_FEW_SHOT_EXAMPLES_GSM8K

FIXED_FEW_SHOT_EXAMPLES_NQ: List[Dict[str, str]] = [
    {
        "question": "who played the original darth vader in star wars?",
        "answer": "David Prowse",
    },
    {
        "question": "what is the currency used in switzerland?",
        "answer": "Swiss franc",
    },
    {
        "question": "when was the treaty of versailles signed?",
        "answer": "June 28, 1919",
    },
]

FIXED_FEW_SHOT_EXAMPLES_CONTEXTEVAL: List[Dict[str, str]] = [
    {
        "context": "The Apollo 11 spacecraft launched from Kennedy Space Center on July 16, 1969, carrying commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin.",
        "question": "Who was the command module pilot on Apollo 11?",
        "answer": "Michael Collins",
    },
    {
        "context": "Photosynthesis occurs in two stages: the light-dependent reactions, which take place in the thylakoid membranes, and the light-independent Calvin cycle, which takes place in the stroma.",
        "question": "Where does the Calvin cycle take place in a plant cell?",
        "answer": "Stroma",
    },
    {
        "context": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, extending from the Arctic Ocean in the north to the Southern Ocean in the south.",
        "question": "Which ocean is the largest and deepest on Earth?",
        "answer": "Pacific Ocean",
    },
]

FIXED_FEW_SHOT_EXAMPLES_CNNDM: List[Dict[str, str]] = [
    {
        "document": (
            "A rare blue diamond has sold at auction in Geneva for a record $48.5 million. "
            "The 12.03-carat 'Blue Moon' diamond was bought by a Hong Kong collector who immediately renamed it "
            "'The Blue Moon of Josephine'. The gemstone was discovered in South Africa in January 2014 and is "
            "considered one of the purest blue diamonds ever discovered."
        ),
        "summary": "A 12.03-carat blue diamond sold for $48.5 million at an auction in Geneva to a Hong Kong collector.",
    },
    {
        "document": (
            "NASA's Curiosity rover has discovered evidence that a large lake once filled Gale Crater on Mars. "
            "Sedimentary rock layers indicate water persisted for millions of years, suggesting Mars may have had "
            "a climate capable of supporting microbial life billions of years ago. Scientists analyzed sediment deposits "
            "at Mount Sharp inside the crater."
        ),
        "summary": "NASA's Curiosity rover found sedimentary evidence that Gale Crater on Mars once hosted a long-standing lake.",
    },
    {
        "document": (
            "Electric vehicle sales reached a historic milestone in Norway, accounting for more than 80% of all new "
            "passenger car sales last year. Government incentives, including tax exemptions and toll discounts, "
            "accelerated the transition as the country aims to end the sale of petrol and diesel cars by 2025."
        ),
        "summary": "Electric vehicles accounted for over 80% of new car sales in Norway, driven by strong government incentives.",
    },
]


# ============================================================
# 4. User Prompt Builders (Standardized Across Paradigms)
# ============================================================

def build_user_prompt(
    strategy: PromptStrategy,
    task_input: str,
    dataset: str = "general",
    context: Optional[str] = None
) -> str:
    """Builds clean, standardized user-turn prompt content."""
    ds = dataset.lower().strip()
    
    # 1. Zero-shot Direct
    if strategy == PromptStrategy.ZERO_SHOT_DIRECT:
        if ds in ("gsm8k", "gsm"):
            return f"Problem:\n{task_input}\n\nAnswer:"
        elif ds in ("cnn_dailymail", "cnn"):
            return f"Document:\n{task_input}\n\nSummary:"
        else:
            return f"Question:\n{task_input}\n\nAnswer:"

    # 2. Zero-shot CoT (Classic Kojima et al.)
    elif strategy == PromptStrategy.ZERO_SHOT_COT:
        if ds in ("gsm8k", "gsm"):
            return f"Problem:\n{task_input}\n\nLet's solve this step by step."
        elif ds in ("cnn_dailymail", "cnn"):
            return f"Document:\n{task_input}\n\nLet's analyze the document key points step by step and write a summary:"
        else:
            return f"Question:\n{task_input}\n\nLet's think step by step."

    # 3. Short CoT (Constrained / Budgeted Reasoning)
    elif strategy == PromptStrategy.SHORT_COT:
        if ds in ("gsm8k", "gsm"):
            return f"Problem:\n{task_input}\n\nBrief reasoning (at most 2 steps):\nAnswer:"
        elif ds in ("cnn_dailymail", "cnn"):
            return f"Document:\n{task_input}\n\nKey bullet points and summary:"
        else:
            return f"Question:\n{task_input}\n\nBrief 1-2 step reasoning and answer:"

    # 4. Long CoT (Detailed / Comprehensive Reasoning)
    elif strategy == PromptStrategy.LONG_COT:
        if ds in ("gsm8k", "gsm"):
            return f"Problem:\n{task_input}\n\nDetailed comprehensive reasoning:"
        elif ds in ("cnn_dailymail", "cnn"):
            return f"Document:\n{task_input}\n\nDetailed narrative analysis and executive summary:"
        else:
            return f"Question:\n{task_input}\n\nDetailed step-by-step reasoning:"

    # 5. Few-Shot (3-Shot In-Context Demonstrations)
    elif strategy == PromptStrategy.FEW_SHOT_3:
        demos: List[str] = []
        if ds in ("gsm8k", "gsm"):
            for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES_GSM8K, start=1):
                demos.append(f"Example {i}\nProblem:\n{ex['problem']}\n\nAnswer:\n{ex['answer']}\n")
            demos.append(f"Target Problem:\n{task_input}\n\nAnswer:")
        elif ds in ("natural_questions", "nq"):
            for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES_NQ, start=1):
                demos.append(f"Example {i}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n")
            demos.append(f"Target Question: {task_input}\nAnswer:")
        elif ds in ("contexteval", "context_eval"):
            for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES_CONTEXTEVAL, start=1):
                demos.append(f"Example {i}\nDocument: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n")
            demos.append(f"Target Question:\n{task_input}\n\nAnswer:")
        elif ds in ("cnn_dailymail", "cnn"):
            for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES_CNNDM, start=1):
                demos.append(f"Example {i}\nArticle:\n{ex['document']}\n\nSummary:\n{ex['summary']}\n")
            demos.append(f"Target Article:\n{task_input}\n\nSummary:")
        else:
            for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES_NQ, start=1):
                demos.append(f"Example {i}\nTask: {ex['question']}\nAnswer: {ex['answer']}\n")
            demos.append(f"Target Task: {task_input}\nAnswer:")
        return "\n".join(demos)

    # 6. RAG / Context Grounded
    elif strategy == PromptStrategy.RAG:
        if ds in ("cnn_dailymail", "cnn"):
            return f"Document:\n{task_input}\n\nSummary:"
        return f"Question:\n{task_input}\n\nAnswer:"

    return f"Task:\n{task_input}\n\nAnswer:"


# ============================================================
# 5. Universal Prompt Registry
# ============================================================

DATASET_PROMPT_REGISTRY: Dict[str, Dict[PromptStrategy, Dict[str, Any]]] = {
    "gsm8k": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_GSM8K_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q, "gsm8k"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.FEW_SHOT_3: {
            "system": SYSTEM_GSM8K_FEW_SHOT_3,
            "builder": lambda q: build_user_prompt(PromptStrategy.FEW_SHOT_3, q, "gsm8k"),
            "uses_demonstrations": True,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_GSM8K_ZERO_SHOT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_COT, q, "gsm8k"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.SHORT_COT: {
            "system": SYSTEM_GSM8K_SHORT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.SHORT_COT, q, "gsm8k"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.LONG_COT: {
            "system": SYSTEM_GSM8K_LONG_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.LONG_COT, q, "gsm8k"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_GSM8K_RAG,
            "builder": lambda q: build_user_prompt(PromptStrategy.RAG, q, "gsm8k"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
    },
    "natural_questions": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q, "natural_questions"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.FEW_SHOT_3: {
            "system": SYSTEM_FEW_SHOT_3,
            "builder": lambda q: build_user_prompt(PromptStrategy.FEW_SHOT_3, q, "natural_questions"),
            "uses_demonstrations": True,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_ZERO_SHOT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_COT, q, "natural_questions"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.SHORT_COT: {
            "system": SYSTEM_SHORT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.SHORT_COT, q, "natural_questions"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.LONG_COT: {
            "system": SYSTEM_LONG_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.LONG_COT, q, "natural_questions"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_RAG,
            "builder": lambda q: build_user_prompt(PromptStrategy.RAG, q, "natural_questions"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
    },
    "contexteval": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_ZERO_SHOT_DIRECT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, q, "contexteval"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
        PromptStrategy.FEW_SHOT_3: {
            "system": SYSTEM_FEW_SHOT_3,
            "builder": lambda q: build_user_prompt(PromptStrategy.FEW_SHOT_3, q, "contexteval"),
            "uses_demonstrations": True,
            "reasoning": False,
            "context_allowed": True,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_ZERO_SHOT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.ZERO_SHOT_COT, q, "contexteval"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": True,
        },
        PromptStrategy.SHORT_COT: {
            "system": SYSTEM_SHORT_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.SHORT_COT, q, "contexteval"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": True,
        },
        PromptStrategy.LONG_COT: {
            "system": SYSTEM_LONG_COT,
            "builder": lambda q: build_user_prompt(PromptStrategy.LONG_COT, q, "contexteval"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": True,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_RAG,
            "builder": lambda q: build_user_prompt(PromptStrategy.RAG, q, "contexteval"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
    },
    "cnn_dailymail": {
        PromptStrategy.ZERO_SHOT_DIRECT: {
            "system": SYSTEM_ZERO_SHOT_DIRECT,
            "builder": lambda t: build_user_prompt(PromptStrategy.ZERO_SHOT_DIRECT, t, "cnn_dailymail"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.FEW_SHOT_3: {
            "system": SYSTEM_FEW_SHOT_3,
            "builder": lambda t: build_user_prompt(PromptStrategy.FEW_SHOT_3, t, "cnn_dailymail"),
            "uses_demonstrations": True,
            "reasoning": False,
            "context_allowed": False,
        },
        PromptStrategy.ZERO_SHOT_COT: {
            "system": SYSTEM_ZERO_SHOT_COT,
            "builder": lambda t: build_user_prompt(PromptStrategy.ZERO_SHOT_COT, t, "cnn_dailymail"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.SHORT_COT: {
            "system": SYSTEM_SHORT_COT,
            "builder": lambda t: build_user_prompt(PromptStrategy.SHORT_COT, t, "cnn_dailymail"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.LONG_COT: {
            "system": SYSTEM_LONG_COT,
            "builder": lambda t: build_user_prompt(PromptStrategy.LONG_COT, t, "cnn_dailymail"),
            "uses_demonstrations": False,
            "reasoning": True,
            "context_allowed": False,
        },
        PromptStrategy.RAG: {
            "system": SYSTEM_RAG,
            "builder": lambda t: build_user_prompt(PromptStrategy.RAG, t, "cnn_dailymail"),
            "uses_demonstrations": False,
            "reasoning": False,
            "context_allowed": True,
        },
    },
}

# Synonyms & Aliases
DATASET_PROMPT_REGISTRY["nq"] = DATASET_PROMPT_REGISTRY["natural_questions"]
DATASET_PROMPT_REGISTRY["cnn"] = DATASET_PROMPT_REGISTRY["cnn_dailymail"]
DATASET_PROMPT_REGISTRY["context_eval"] = DATASET_PROMPT_REGISTRY["contexteval"]
PROMPT_REGISTRY = DATASET_PROMPT_REGISTRY["gsm8k"]


# ============================================================
# 6. Universal Prompt Formatter
# ============================================================

def _normalize_strategy(strategy: Union[PromptStrategy, str]) -> PromptStrategy:
    """Safely coerces string strategy representations into PromptStrategy Enum."""
    if isinstance(strategy, PromptStrategy):
        return strategy
    s_val = str(strategy).strip().lower()
    if s_val.startswith("rag_top_") or s_val == "rag":
        return PromptStrategy.RAG
    if s_val.startswith("ctx_"):
        return PromptStrategy.ZERO_SHOT_DIRECT
    try:
        return PromptStrategy(s_val)
    except ValueError:
        return PromptStrategy.ZERO_SHOT_DIRECT


def format_prompt(
    dataset: str,
    strategy: Union[PromptStrategy, str],
    input_text: str,
    context: Optional[str] = None,
    allow_context: bool = False
) -> List[Dict[str, str]]:
    """Universal prompt formatter for all benchmark datasets and task families."""
    dataset_key = dataset.lower().strip()
    if dataset_key not in DATASET_PROMPT_REGISTRY:
        dataset_key = "gsm8k"

    strat_enum = _normalize_strategy(strategy)
    registry = DATASET_PROMPT_REGISTRY[dataset_key]

    if strat_enum not in registry:
        strat_enum = PromptStrategy.ZERO_SHOT_DIRECT

    config = registry[strat_enum]

    user_content = config["builder"](input_text)
    if context:
        user_content = f"Context:\n{context}\n\n{user_content}"

    return [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": user_content}
    ]


def format_gsm8k_prompt(
    strategy: Union[PromptStrategy, str],
    question: str,
    context: Optional[str] = None,
    allow_context: bool = False
) -> List[Dict[str, str]]:
    """Helper for GSM8K prompt formatting."""
    return format_prompt(
        dataset="gsm8k",
        strategy=strategy,
        input_text=question,
        context=context,
        allow_context=allow_context
    )


# ============================================================
# 7. Metadata & Reproducibility Hashing
# ============================================================

def get_prompt_metadata(
    strategy: Union[PromptStrategy, str],
    dataset: str = "gsm8k"
) -> Dict[str, Any]:
    """Return metadata describing the experimental prompt condition."""
    dataset_key = dataset.lower().strip()
    strat_enum = _normalize_strategy(strategy)
    config = DATASET_PROMPT_REGISTRY.get(dataset_key, DATASET_PROMPT_REGISTRY["gsm8k"])[strat_enum]

    demos_count = 0
    if config["uses_demonstrations"]:
        if dataset_key in ("gsm8k", "gsm"):
            demos_count = len(FIXED_FEW_SHOT_EXAMPLES_GSM8K)
        elif dataset_key in ("natural_questions", "nq"):
            demos_count = len(FIXED_FEW_SHOT_EXAMPLES_NQ)
        elif dataset_key in ("contexteval", "context_eval"):
            demos_count = len(FIXED_FEW_SHOT_EXAMPLES_CONTEXTEVAL)
        elif dataset_key in ("cnn_dailymail", "cnn"):
            demos_count = len(FIXED_FEW_SHOT_EXAMPLES_CNNDM)

    return {
        "dataset": dataset_key,
        "strategy": strat_enum.value,
        "uses_demonstrations": config["uses_demonstrations"],
        "reasoning": config["reasoning"],
        "context_allowed": config["context_allowed"],
        "num_demonstrations": demos_count
    }


def get_prompt_hash(
    prompt_version: str = "promptenergy_v2.1",
    dataset: str = "gsm8k"
) -> str:
    """Compute a deterministic SHA-256 hash of the prompt registry for scientific reproducibility."""
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
    if dataset_key in ("gsm8k", "gsm"):
        payload["few_shot_examples"] = FIXED_FEW_SHOT_EXAMPLES_GSM8K
    elif dataset_key in ("natural_questions", "nq"):
        payload["few_shot_examples"] = FIXED_FEW_SHOT_EXAMPLES_NQ
    elif dataset_key in ("contexteval", "context_eval"):
        payload["few_shot_examples"] = FIXED_FEW_SHOT_EXAMPLES_CONTEXTEVAL
    elif dataset_key in ("cnn_dailymail", "cnn"):
        payload["few_shot_examples"] = FIXED_FEW_SHOT_EXAMPLES_CNNDM

    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_prompt_integrity() -> None:
    """Validates prompt integrity across all 4 datasets."""
    assert len(FIXED_FEW_SHOT_EXAMPLES_GSM8K) == 3
    for ex in FIXED_FEW_SHOT_EXAMPLES_GSM8K:
        assert "problem" in ex and "answer" in ex and ex["answer"].startswith("#### ")
    gsm_reg = DATASET_PROMPT_REGISTRY["gsm8k"]
    assert gsm_reg[PromptStrategy.ZERO_SHOT_DIRECT]["uses_demonstrations"] is False
    assert gsm_reg[PromptStrategy.ZERO_SHOT_COT]["uses_demonstrations"] is False
    assert gsm_reg[PromptStrategy.FEW_SHOT_3]["uses_demonstrations"] is True

    assert len(FIXED_FEW_SHOT_EXAMPLES_NQ) == 3
    assert len(FIXED_FEW_SHOT_EXAMPLES_CONTEXTEVAL) == 3
    assert len(FIXED_FEW_SHOT_EXAMPLES_CNNDM) == 3
