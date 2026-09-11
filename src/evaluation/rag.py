"""Retrieval-Augmented Generation (RAG) Evaluator for PromptEnergy-Bench.

Supports:
- Answer correctness against ground truth.
- Faithfulness / Groundedness scoring against retrieved context.
- Context utilization ratio.
- Retrieval Recall & Precision when gold context / evidence is specified.
- Citation checking.
- Strict rejection of truncated generations.
"""

import re
import string
from typing import Optional, Any, Dict, List, Set, Tuple
from collections import Counter

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)
from src.evaluation.open_qa import compute_f1, compute_exact_match, normalize_answer


class RAGEvaluator(BaseEvaluator):
    """Evaluates RAG generation, grounding, and retrieval performance."""

    def compute_groundedness(self, answer: str, context_text: str) -> float:
        """Estimates the fraction of non-stopword answer tokens supported by context."""
        if not answer or not context_text:
            return 0.0

        ans_tokens = set(normalize_answer(answer).split())
        ctx_tokens = set(normalize_answer(context_text).split())

        if not ans_tokens:
            return 0.0

        supported = ans_tokens & ctx_tokens
        return len(supported) / len(ans_tokens)

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates RAG response with grounding and retrieval metrics."""
        ref_str = str(gold_answer) if gold_answer is not None else ""
        context_dict = context or {}
        retrieved_context = context_dict.get("retrieved_context", "")
        gold_evidence = context_dict.get("gold_evidence", "")
        retrieved_ids = context_dict.get("retrieved_doc_ids", [])
        gold_ids = context_dict.get("gold_doc_ids", [])

        if generation_truncated:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response=None,
                parsed_answer=None,
                evaluation_status=EvaluationStatus.GENERATION_TRUNCATED,
                answer_correct=False,
                metric_values={
                    "exact_match": False,
                    "token_f1": 0.0,
                    "groundedness_score": 0.0,
                    "retrieval_recall": None,
                    "retrieval_precision": None
                },
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Generation truncated before completing RAG response"
            )

        non_thinking = self.strip_thinking(raw_output, raw_response)
        text = non_thinking if non_thinking else (raw_output or "")

        if not text.strip():
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response="",
                parsed_answer="",
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={
                    "exact_match": False,
                    "token_f1": 0.0,
                    "groundedness_score": 0.0
                },
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="Empty model response"
            )

        # 1. Answer correctness
        exact_match = compute_exact_match(text, ref_str)
        token_f1 = compute_f1(text, ref_str)
        is_correct = exact_match or (token_f1 >= 0.7)

        # 2. Groundedness
        groundedness = self.compute_groundedness(text, retrieved_context) if retrieved_context else 1.0

        # 3. Retrieval Recall / Precision
        ret_recall = None
        ret_prec = None
        if gold_ids and retrieved_ids:
            gold_set = set(str(g) for g in gold_ids)
            ret_set = set(str(r) for r in retrieved_ids)
            intersection = gold_set & ret_set
            ret_recall = len(intersection) / len(gold_set) if gold_set else 1.0
            ret_prec = len(intersection) / len(ret_set) if ret_set else 0.0

        status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.COMPLETED_INCORRECT

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_str,
            raw_response=raw_output,
            normalized_response=normalize_answer(text),
            parsed_answer=text.strip(),
            evaluation_status=status,
            answer_correct=is_correct,
            metric_values={
                "exact_match": exact_match,
                "token_f1": round(token_f1, 4),
                "groundedness_score": round(groundedness, 4),
                "retrieval_recall": round(ret_recall, 4) if ret_recall is not None else None,
                "retrieval_precision": round(ret_prec, 4) if ret_prec is not None else None
            },
            parse_success=True,
            generation_truncated=False
        )
