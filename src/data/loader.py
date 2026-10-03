"""Universal Benchmark Dataset Loader for PromptEnergy-Bench.

Supports standardized ingestion and slicing for all 4 benchmark datasets:
1. GSM8K (Mathematical Reasoning)
2. Natural Questions (Knowledge-Intensive QA & RAG)
3. ContextEval (Long-Context QA & Scaling)
4. CNN/DailyMail (Abstractive Summarization)
"""

import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any

from src.data.gsm8k import load_gsm8k, GSM8KRecord, EVAL_SAMPLE_SIZE, EVAL_SEED


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

    @property
    def solution(self) -> str:
        if self.metadata and isinstance(self.metadata, dict) and "solution" in self.metadata:
            return str(self.metadata["solution"])
        if self.context:
            return self.context
        return self.target_text or ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_benchmark_dataset(
    dataset_name: str,
    split: str = "test",
    eval_size: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
    seed: int = EVAL_SEED
) -> List[BenchmarkRecord]:
    """Loads and standardizes records for any registered benchmark dataset.
    
    For evaluation splits ('test', 'validation', 'val'), deterministically samples
    up to EVAL_SAMPLE_SIZE (1,000) records using a fixed seed (EVAL_SEED=42).

    Args:
        dataset_name: 'gsm8k', 'natural_questions', 'contexteval', 'cnn_dailymail'.
        split: Target split name ('test', 'train', 'validation').
        eval_size: Slicing count or 'full' (capped at EVAL_SAMPLE_SIZE=1000).
        dataset_dir: Optional custom dataset root path.
        seed: Random seed for deterministic sample slicing (default: 42).

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

    # Specific dataset handling for GSM8K
    if key == "gsm8k":
        gsm_records = load_gsm8k(split=split, eval_size=eval_size, dataset_dir=root, seed=seed)
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
        # Fallback to any existing split in the dataset directory
        found_path = None
        for alt_split in ["test", "validation", "val", "train"]:
            alt_path = os.path.join(root, f"{alt_split}.jsonl")
            if os.path.exists(alt_path):
                found_path = alt_path
                split = alt_split
                break
        if found_path:
            file_path = found_path
        else:
            raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Ingest records
    raw_records: List[BenchmarkRecord] = []
    # For large background files like NQ train when used as test fallback, cap scan to 5000 lines
    max_scan = 5000 if (key in ("natural_questions", "nq") and split == "train") else None

    with open(file_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_scan is not None and idx >= max_scan:
                break
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

            raw_records.append(BenchmarkRecord(
                id=rec_id,
                dataset=key,
                split=split,
                input_text=input_text.strip(),
                target_text=target_text.strip() if target_text else None,
                context=context.strip() if context else None,
                metadata=data
            ))

    total_count = len(raw_records)

    # For evaluation splits: deterministically sample up to EVAL_SAMPLE_SIZE (1,000)
    is_eval_split = split in ("test", "validation", "val") or (split == "train" and eval_size is not None and int(eval_size) if str(eval_size).isdigit() else True)
    
    if is_eval_split and split != "train":
        if total_count > EVAL_SAMPLE_SIZE:
            rng = random.Random(seed)
            sampled_indices = sorted(rng.sample(range(total_count), EVAL_SAMPLE_SIZE))
            records = [raw_records[i] for i in sampled_indices]
        else:
            records = raw_records
        
        dataset_display = key.upper().replace("_", " ")
        print(f"{dataset_display}: loaded {len(records)} / {total_count} {split} samples (seed={seed})")
    elif split == "train" and key in ("natural_questions", "nq"):
        # For NQ where only train.jsonl exists, sample 1000 for evaluation if used as evaluation
        if total_count > EVAL_SAMPLE_SIZE:
            rng = random.Random(seed)
            sampled_indices = sorted(rng.sample(range(total_count), EVAL_SAMPLE_SIZE))
            records = [raw_records[i] for i in sampled_indices]
        else:
            records = raw_records
        dataset_display = key.upper().replace("_", " ")
        print(f"{dataset_display}: loaded {len(records)} / {total_count} evaluation samples (seed={seed})")
    else:
        records = raw_records

    # Apply explicit eval_size slice if specified (e.g. eval_size=10 for quick testing)
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
