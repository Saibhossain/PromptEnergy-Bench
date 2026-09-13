"""Controlled Context Builder for Experiment 2.

Constructs controlled context from GSM8K TRAIN examples for a fixed target GSM8K TEST question.
Supports:
1. Relevant context: mathematically aligned training examples.
2. Distractor context: semantically distant training examples.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Any, Dict
from src.data.gsm8k import GSM8KRecord, load_gsm8k


@dataclass
class ScaledContext:
    context_type: str  # 'relevant' or 'distractor'
    target_context_tokens: int
    actual_context_tokens: int
    context_text: str
    context_document_ids: List[str]


def estimate_tokens(text: str) -> int:
    """Fast, reproducible token count estimation based on whitespace + punctuation splitting.
    
    Approximately 1 token per 3.8 characters or 0.75 words, consistent across platforms.
    """
    if not text:
        return 0
    # Standard heuristic: split on whitespace and punctuation tokens
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, int(len(words) * 1.1))


class ContextBuilder:
    def __init__(self, train_records: Optional[List[GSM8KRecord]] = None, dataset_dir: str = "datasets/gsm8k"):
        if train_records is None:
            self.train_records = load_gsm8k(split="train", dataset_dir=dataset_dir)
        else:
            self.train_records = train_records
        
        # Precompute keywords for each training record for relevance scoring
        self.doc_keywords: List[Set[str]] = [
            self._extract_keywords(rec.question) for rec in self.train_records
        ]

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # Remove common stop words
        stops = {
            "the", "and", "that", "have", "for", "not", "with", "you", "this",
            "but", "his", "from", "they", "she", "which", "how", "many", "does",
            "much", "what", "then", "each", "every", "per", "all"
        }
        return {w for w in words if w not in stops}

    def _rank_by_relevance(self, target_question: str, exclude_id: Optional[str] = None) -> List[int]:
        target_kws = self._extract_keywords(target_question)
        scores = []
        for idx, kws in enumerate(self.doc_keywords):
            if exclude_id and self.train_records[idx].id == exclude_id:
                continue
            overlap = len(target_kws.intersection(kws)) if target_kws else 0
            scores.append((overlap, idx))

        # Higher overlap first
        scores.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scores]

    def _rank_by_distractor(self, target_question: str, exclude_id: Optional[str] = None) -> List[int]:
        target_kws = self._extract_keywords(target_question)
        scores = []
        for idx, kws in enumerate(self.doc_keywords):
            if exclude_id and self.train_records[idx].id == exclude_id:
                continue
            overlap = len(target_kws.intersection(kws)) if target_kws else 0
            scores.append((overlap, idx))

        # Lowest overlap first (zero overlap prioritized)
        scores.sort(key=lambda x: x[0])
        return [idx for _, idx in scores]

    def build_context(
        self,
        target_record: Optional[Any] = None,
        context_type: str = "relevant",
        target_tokens: int = 0,
        exclude_id: Optional[str] = None,
        target_question: Optional[str] = None
    ) -> ScaledContext:
        """Builds a context string of approximately target_tokens length.
        
        Args:
            target_record: Target sample (GSM8KRecord) containing question and ID.
            context_type: 'relevant' (default) or 'distractor'.
            target_tokens: Desired token length (e.g., 0, 512, 1024, 2048, 4096, 8192).
            exclude_id: Optional ID to exclude from reference context (prevents contamination).
            target_question: Optional direct query string if target_record is omitted.
        """
        # Extract target question and exclude_id
        q_text = ""
        ex_id = exclude_id
        if target_record is not None:
            if hasattr(target_record, "question"):
                q_text = target_record.question
            elif isinstance(target_record, dict):
                q_text = target_record.get("question", "")
            if ex_id is None:
                if hasattr(target_record, "id"):
                    ex_id = target_record.id
                elif isinstance(target_record, dict):
                    ex_id = target_record.get("id")
        elif target_question:
            q_text = target_question

        if context_type == "relevant":
            indices = self._rank_by_relevance(q_text, exclude_id=ex_id)
        elif context_type == "distractor":
            indices = self._rank_by_distractor(q_text, exclude_id=ex_id)
        else:
            raise ValueError(f"Unknown context_type: {context_type}. Expected 'relevant' or 'distractor'.")

        if target_tokens <= 0:
            return ScaledContext(
                context_type=context_type,
                target_context_tokens=0,
                actual_context_tokens=0,
                context_text="",
                context_document_ids=[]
            )

        selected_ids: List[str] = []
        snippets: List[str] = []
        current_tokens = 0

        header = "Background Context (Mathematical Reference Examples):\n"
        snippets.append(header)
        current_tokens += estimate_tokens(header)

        for idx in indices:
            rec = self.train_records[idx]
            # Format reference example
            item = f"[Reference Problem {rec.id}]:\nProblem: {rec.question}\nSolution & Reasoning: {rec.solution}\nResult: {rec.answer}\n\n"
            item_tokens = estimate_tokens(item)

            if current_tokens + item_tokens > target_tokens and len(selected_ids) > 0:
                break

            snippets.append(item)
            selected_ids.append(rec.id)
            current_tokens += item_tokens

            if current_tokens >= target_tokens:
                break

        full_context = "".join(snippets)
        actual_tokens = estimate_tokens(full_context)

        return ScaledContext(
            context_type=context_type,
            target_context_tokens=target_tokens,
            actual_context_tokens=actual_tokens,
            context_text=full_context,
            context_document_ids=selected_ids
        )
