"""Universal Benchmark Dataset Loader for PromptEnergy-Bench.

Supports standardized ingestion and slicing for all 4 benchmark datasets:
1. GSM8K (Mathematical Reasoning)
2. Natural Questions (Knowledge-Intensive QA & RAG)
3. ContextEval (Long-Context QA & Scaling)
4. CNN/DailyMail (Abstractive Summarization)
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any

from src.data.gsm8k import load_gsm8k, GSM8KRecord


@dataclass(frozen=True)
class BenchmarkRecord:
    id: str
    dataset: str
    split: str
    input_text: str
    target_text: Optional[str]
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Backward-compatible property accessors
    @property
    def question(self) -> str:
        return self.input_text

    @property
    def answer(self) -> str:
        return self.target_text or ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_benchmark_dataset(
    dataset_name: str,
    split: str = "test",
    eval_size: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
    seed: int = 42
) -> List[BenchmarkRecord]:
    """Loads and standardizes records for any registered benchmark dataset.

    Args:
        dataset_name: 'gsm8k', 'natural_questions', 'contexteval', 'cnn_dailymail'.
        split: Target split name ('test', 'train', 'validation').
        eval_size: Slicing count or 'full'.
        dataset_dir: Optional custom dataset root path.
        seed: Random seed for deterministic sample slicing.

    Returns:
        List of BenchmarkRecord instances with standardized input_text and target_text.
    """
    key = dataset_name.lower().strip()
    root = dataset_dir or os.path.join("datasets", key)

    if not os.path.exists(root):
        # Fallback path checking
        alt_root = os.path.join("datasets", key.replace("_", "-"))
        if os.path.exists(alt_root):
            root = alt_root
        else:
            raise FileNotFoundError(f"Dataset directory not found: {root}")

    # Specific dataset handling
    if key == "gsm8k":
        gsm_records = load_gsm8k(split=split, eval_size=eval_size, dataset_dir=root)
        return [
            BenchmarkRecord(
                id=r.id,
                dataset="gsm8k",
                split=split,
                input_text=r.question,
                target_text=r.answer,
                metadata={"solution": r.solution, "raw_answer": r.raw_answer}
            )
            for r in gsm_records
        ]

    # Generalized JSONL loading
    file_path = os.path.join(root, f"{split}.jsonl")
    if not os.path.exists(file_path):
        # If test split was requested but only train exists (e.g. NQ pair)
        if split == "test" and os.path.exists(os.path.join(root, "train.jsonl")):
            file_path = os.path.join(root, "train.jsonl")
            split = "train"
        else:
            raise FileNotFoundError(f"Data file not found at: {file_path}")

    records: List[BenchmarkRecord] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_id = f"{key}_{split}_{idx:05d}"
            input_text = ""
            target_text = None
            context = None

            if key in ("natural_questions", "nq"):
                input_text = data.get("query") or data.get("question") or ""
                target_text = data.get("answer") or data.get("passage") or ""
            elif key in ("contexteval", "context_eval"):
                input_text = data.get("query") or data.get("prompt") or ""
                context = data.get("context")
                target_text = data.get("answer") or data.get("reference")
            elif key in ("cnn_dailymail", "cnn"):
                input_text = data.get("article") or ""
                target_text = data.get("highlights") or ""
                rec_id = data.get("id") or rec_id
            else:
                input_text = data.get("input") or data.get("question") or data.get("text") or ""
                target_text = data.get("target") or data.get("answer") or data.get("output")

            records.append(BenchmarkRecord(
                id=rec_id,
                dataset=key,
                split=split,
                input_text=input_text.strip(),
                target_text=target_text.strip() if target_text else None,
                context=context.strip() if context else None,
                metadata=data
            ))

    # Deterministic slice if eval_size is passed
    if eval_size is not None and str(eval_size).lower() != "full":
        try:
            limit = int(eval_size)
            if limit < 0:
                raise ValueError("eval_size must be positive.")
            records = records[:limit]
        except ValueError:
            pass

    return records
