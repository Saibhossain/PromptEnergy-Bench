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


import heapq

class ContextBuilder:
    def __init__(self, train_records: Optional[List[GSM8KRecord]] = None, dataset_dir: str = "datasets/gsm8k"):
        if train_records is None:
            self.train_records = load_gsm8k(split="train", dataset_dir=dataset_dir)
        else:
            self.train_records = train_records
        
        # Precompute keywords and inverted keyword index for rapid sparse lookup
        self.doc_keywords: List[Set[str]] = []
        self.keyword_to_docs: Dict[str, List[int]] = {}
        
        for idx, rec in enumerate(self.train_records):
            kws = self._extract_keywords(getattr(rec, "question", getattr(rec, "input_text", "")))
            self.doc_keywords.append(kws)
            for kw in kws:
                if kw not in self.keyword_to_docs:
                    self.keyword_to_docs[kw] = []
                self.keyword_to_docs[kw].append(idx)

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

    def _rank_by_relevance(self, target_question: str, exclude_id: Optional[str] = None, max_candidates: int = 150) -> List[int]:
        target_kws = self._extract_keywords(target_question)
        if not target_kws:
            return [i for i, r in enumerate(self.train_records) if not (exclude_id and r.id == exclude_id)]

        # Sparse scoring via inverted keyword index
        scores: Dict[int, int] = {}
        for kw in target_kws:
            if kw in self.keyword_to_docs:
                for idx in self.keyword_to_docs[kw]:
                    if exclude_id and self.train_records[idx].id == exclude_id:
                        continue
                    scores[idx] = scores.get(idx, 0) + 1

        if not scores:
            return [i for i, r in enumerate(self.train_records) if not (exclude_id and r.id == exclude_id)]

        # Highest overlap first
        sorted_candidates = heapq.nlargest(min(len(scores), max_candidates), scores.keys(), key=lambda i: scores[i])
        
        # If additional documents are needed, append remaining documents
        if len(sorted_candidates) < max_candidates:
            seen = set(sorted_candidates)
            for i, r in enumerate(self.train_records):
                if exclude_id and r.id == exclude_id:
                    continue
                if i not in seen:
                    sorted_candidates.append(i)
                    if len(sorted_candidates) >= max_candidates:
                        break
        return sorted_candidates

    def _rank_by_distractor(self, target_question: str, exclude_id: Optional[str] = None, max_candidates: int = 150) -> List[int]:
        target_kws = self._extract_keywords(target_question)
        if not target_kws:
            return [i for i, r in enumerate(self.train_records) if not (exclude_id and r.id == exclude_id)]

        # Collect documents that share at least one keyword
        matching_docs: Set[int] = set()
        for kw in target_kws:
            if kw in self.keyword_to_docs:
                matching_docs.update(self.keyword_to_docs[kw])

        # Distractors are documents with zero keyword overlap
        distractors: List[int] = []
        for idx, rec in enumerate(self.train_records):
            if exclude_id and rec.id == exclude_id:
                continue
            if idx not in matching_docs:
                distractors.append(idx)
                if len(distractors) >= max_candidates:
                    break

        # Fallback to remaining if more documents are needed
        if len(distractors) < max_candidates:
            for idx, rec in enumerate(self.train_records):
                if exclude_id and rec.id == exclude_id:
                    continue
                if idx not in distractors:
                    distractors.append(idx)
                    if len(distractors) >= max_candidates:
                        break

        return distractors

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

        is_math = False
        if self.train_records:
            first_rec = self.train_records[0]
            ds = getattr(first_rec, "dataset", "")
            if ds in ("gsm8k", "math", "svamp"):
                is_math = True

        header = f"Background Context ({'Mathematical Reference Examples' if is_math else 'Reference Documents'}):\n"
        snippets.append(header)
        current_tokens += estimate_tokens(header)

        for idx in indices:
            rec = self.train_records[idx]
            q = getattr(rec, "question", getattr(rec, "input_text", ""))
            sol = getattr(rec, "solution", getattr(rec, "context", ""))
            ans = getattr(rec, "answer", getattr(rec, "target_text", ""))
            r_id = getattr(rec, "id", str(idx))
            
            if is_math:
                if sol and sol != ans:
                    item = f"[Reference Item {r_id}]:\nInput: {q}\nContent/Reasoning: {sol}\nTarget: {ans}\n\n"
                else:
                    item = f"[Reference Item {r_id}]:\nInput: {q}\nTarget: {ans}\n\n"
            else:
                if sol and sol != ans:
                    item = f"[Reference Document {r_id}]:\nQuery: {q}\nPassage: {sol}\n\n"
                elif ans:
                    item = f"[Reference Document {r_id}]:\nContent: {ans}\n\n"
                else:
                    item = f"[Reference Document {r_id}]:\nText: {q}\n\n"
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
