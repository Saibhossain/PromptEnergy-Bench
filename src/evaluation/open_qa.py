"""Open-Ended Question Answering Evaluator for PromptEnergy-Bench.

Supports:
- SQuAD standard normalization (lowercase, remove punctuation, remove articles 'a', 'an', 'the', whitespace collapse).
- Exact Match (EM) and Token-level F1 calculation.
- Multi-reference evaluation with policies ('max', 'all', 'first').
- Strict rejection of truncated generations.
"""

import re
import string
from typing import Optional, Any, Dict, List, Set, Tuple, Union
from collections import Counter

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 between prediction and single ground truth string."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Computes normalized exact match."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


class OpenQAEvaluator(BaseEvaluator):
    """Evaluates open-ended question answering tasks (NQ, TriviaQA, SQuAD)."""

    def extract_answer(
        self,
        raw_output: str,
        raw_response: Optional[str] = None
    ) -> str:
        """Extracts candidate answer string from output."""
        non_thinking = self.strip_thinking(raw_output, raw_response)
        target = non_thinking if non_thinking else raw_output

        # Check for explicit answer phrase
        phrase_match = re.search(r"(?:the\s+answer\s+is|answer\s*[:=]|final\s+answer\s*[:=])\s*(.+)", target, re.IGNORECASE)
        if phrase_match:
            # Take the rest of the line or first sentence
            cand = phrase_match.group(1).split("\n")[0].strip()
            if cand:
                return cand

        # Otherwise return non-thinking text or first line
        lines = [l.strip() for l in target.split("\n") if l.strip()]
        return lines[0] if lines else target.strip()

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Union[str, List[str]],
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates model prediction against single or multiple reference answers."""
        # Standardize references to list
        if isinstance(gold_answer, list):
            references = [str(a) for a in gold_answer if a is not None]
        else:
            references = [str(gold_answer)] if gold_answer is not None else [""]

        if not references:
            references = [""]

        ref_repr = references[0] if len(references) == 1 else references

        if generation_truncated:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_repr,
                raw_response=raw_output,
                normalized_response=None,
                parsed_answer=None,
                evaluation_status=EvaluationStatus.GENERATION_TRUNCATED,
                answer_correct=False,
                metric_values={"exact_match": False, "token_f1": 0.0},
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Generation truncated before completing answer"
            )

        if not raw_output or not str(raw_output).strip():
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_repr,
                raw_response=raw_output,
                normalized_response="",
                parsed_answer=None,
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"exact_match": False, "token_f1": 0.0},
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="Empty model response"
            )

        extracted = self.extract_answer(raw_output, raw_response)
        norm_pred = normalize_answer(extracted)

        # Multi-reference scoring
        em_scores = [compute_exact_match(extracted, ref) for ref in references]
        f1_scores = [compute_f1(extracted, ref) for ref in references]

        policy = self.config.multiple_reference_policy
        if policy == "first":
            exact_match = em_scores[0]
            token_f1 = f1_scores[0]
        elif policy == "all":
            exact_match = all(em_scores)
            token_f1 = sum(f1_scores) / len(f1_scores)
        else:  # "max" default
            exact_match = any(em_scores)
            token_f1 = max(f1_scores)

        # For Open-QA, consider answer correct if exact match or high F1 (>= 0.8)
        is_correct = exact_match or (token_f1 >= 0.8)
        status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.COMPLETED_INCORRECT

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_repr,
            raw_response=raw_output,
            normalized_response=norm_pred,
            parsed_answer=extracted,
            evaluation_status=status,
            answer_correct=is_correct,
            metric_values={
                "exact_match": exact_match,
                "token_f1": round(token_f1, 4),
                "best_reference": references[f1_scores.index(max(f1_scores))] if references else ""
            },
            parse_success=True,
            generation_truncated=False
        )
