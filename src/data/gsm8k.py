"""GSM8K Dataset Loader and Normalizer.

Ensures strict separation between:
- Evaluation split (datasets/gsm8k/test.jsonl)
- Context / Few-shot / Retrieval split (datasets/gsm8k/train.jsonl)
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Union


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
    dataset_dir: str = "datasets/gsm8k"
) -> List[GSM8KRecord]:
    """Loads GSM8K dataset records.
    
    Args:
        split: 'test' (evaluation) or 'train' (few-shot/context source).
        eval_size: Number of records to evaluate (e.g., 50) or 'full' (all records).
        dataset_dir: Directory containing jsonl files.
    
    Returns:
        List of GSM8KRecord instances.
    """
    if split not in ("test", "train"):
        raise ValueError(f"Invalid split '{split}'. Must be 'test' or 'train'.")

    file_path = os.path.join(dataset_dir, f"{split}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GSM8K file not found at: {file_path}")

    records: List[GSM8KRecord] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = parse_gsm8k_line(line, split, idx)
            if rec is not None:
                records.append(rec)

    if eval_size is not None and str(eval_size).lower() != "full":
        try:
            limit = int(eval_size)
            if limit < 0:
                raise ValueError("eval_size must be positive.")
            records = records[:limit]
        except ValueError as e:
            raise ValueError(f"Invalid eval_size: {eval_size}") from e

    return records
