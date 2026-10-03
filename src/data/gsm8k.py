"""GSM8K Dataset Loader and Normalizer.

Ensures strict separation between:
- Evaluation split (datasets/gsm8k/test.jsonl)
- Context / Few-shot / Retrieval split (datasets/gsm8k/train.jsonl)
"""

import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

# Global Evaluation Sampling Constants
EVAL_SAMPLE_SIZE = 1000
EVAL_SEED = 42


@dataclass(frozen=True)
class GSM8KRecord:
    id: str
    question: str
    answer: str
    solution: str
    raw_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_gold_answer(raw_answer: str) -> str:
    """Extracts and normalizes the final answer following '####' in GSM8K."""
    if "####" in raw_answer:
        ans = raw_answer.split("####")[-1].strip()
    else:
        # Fallback to the last line or tokens
        ans = raw_answer.strip().split("\n")[-1].strip()
    
    # Remove commas in numbers (e.g., 70,000 -> 70000) and currency symbols
    ans = re.sub(r"[,$]", "", ans).strip()
    return ans


def parse_gsm8k_line(line: str, split: str, index: int) -> Optional[GSM8KRecord]:
    """Parses a single line from a GSM8K jsonl file into a normalized GSM8KRecord."""
    line = line.strip()
    if not line:
        return None
    data = json.loads(line)
    question = data.get("question", "").strip()
    raw_answer = data.get("answer", "").strip()

    if "####" in raw_answer:
        parts = raw_answer.split("####")
        solution = parts[0].strip()
        answer = normalize_gold_answer(parts[1])
    else:
        solution = raw_answer
        answer = normalize_gold_answer(raw_answer)

    record_id = f"gsm8k_{split}_{index:04d}"
    return GSM8KRecord(
        id=record_id,
        question=question,
        answer=answer,
        solution=solution,
        raw_answer=raw_answer
    )


def load_gsm8k(
    split: str = "test",
    eval_size: Optional[Union[int, str]] = None,
    dataset_dir: str = "datasets/gsm8k",
    seed: int = EVAL_SEED
) -> List[GSM8KRecord]:
    """Loads GSM8K dataset records with deterministic 1,000 sample evaluation capping.
    
    Args:
        split: 'test' (evaluation) or 'train' (few-shot/context source).
        eval_size: Number of records to evaluate (e.g., 50) or 'full' (capped at EVAL_SAMPLE_SIZE=1000).
        dataset_dir: Directory containing jsonl files.
        seed: Random seed for deterministic sample selection.
    
    Returns:
        List of GSM8KRecord instances.
    """
    if split not in ("test", "train"):
        raise ValueError(f"Invalid split '{split}'. Must be 'test' or 'train'.")

    file_path = os.path.join(dataset_dir, f"{split}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GSM8K file not found at: {file_path}")

    raw_records: List[GSM8KRecord] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = parse_gsm8k_line(line, split, idx)
            if rec is not None:
                raw_records.append(rec)

    total_count = len(raw_records)

    # For evaluation (test split): deterministically cap to EVAL_SAMPLE_SIZE (1,000)
    if split == "test":
        if total_count > EVAL_SAMPLE_SIZE:
            rng = random.Random(seed)
            sampled_indices = sorted(rng.sample(range(total_count), EVAL_SAMPLE_SIZE))
            records = [raw_records[i] for i in sampled_indices]
        else:
            records = raw_records
        print(f"GSM8K: loaded {len(records)} / {total_count} test samples (seed={seed})")
    else:
        records = raw_records

    # Apply explicit eval_size slice if specified (e.g. eval_size=10 for fast debugging)
    if eval_size is not None:
        s_eval = str(eval_size).strip().lower()
        if s_eval not in ("full", "all", "max"):
            limit = None
            if s_eval.endswith("k"):
                try:
                    limit = int(float(s_eval[:-1]) * 1000)
                except ValueError:
                    pass
            elif s_eval.isdigit():
                limit = int(s_eval)
            if limit is not None and limit > 0:
                records = records[:limit]

    return records
