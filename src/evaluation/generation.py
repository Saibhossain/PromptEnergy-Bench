"""Generation and Summarization Evaluator for PromptEnergy-Bench.

Supports:
- Summarization and freeform generation benchmarks (CNN/DailyMail, XSum, WMT, etc.).
- Pure-Python ROUGE-1, ROUGE-2, ROUGE-L (LCS-based F1/Precision/Recall).
- Sentence BLEU calculation with brevity penalty.
- Repetition detection (distinct-1, distinct-2) and length metrics.
- Strict rejection of truncated generations.
"""

import math
import re
from typing import Optional, Any, Dict, List, Set, Tuple
from collections import Counter

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extracts n-grams from list of tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int = 1) -> Tuple[float, float, float]:
    """Computes ROUGE-N precision, recall, and F1."""
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_get_ngrams(ref_tokens, n))

    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_count = max(len(pred_tokens) - n + 1, 0)
    ref_count = max(len(ref_tokens) - n + 1, 0)

    if pred_count == 0 or ref_count == 0 or overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / pred_count
    recall = overlap / ref_count
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Computes the length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    """Computes ROUGE-L (LCS) precision, recall, and F1."""
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0, 0.0, 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_sentence_bleu(pred_tokens: List[str], ref_tokens: List[str], max_n: int = 4) -> float:
    """Computes sentence-level BLEU score with standard smoothing."""
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    c = len(pred_tokens)
    r = len(ref_tokens)
    if c == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1.0 - r / c)

    weights = [1.0 / max_n] * max_n
    precisions = []

    for n in range(1, max_n + 1):
        pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
        ref_ngrams = Counter(_get_ngrams(ref_tokens, n))
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = max(len(pred_tokens) - n + 1, 0)
        # Smoothing: add 1 / (total + 1) if 0
        p = (overlap + 0.1) / (total + 0.1) if total > 0 else 0.0
        precisions.append(p)

    s = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    return bp * math.exp(s)


class GenerationEvaluator(BaseEvaluator):
    """Evaluates text generation and summarization datasets."""

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates generated summary/text against reference."""
        ref_str = str(gold_answer) if gold_answer is not None else ""

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
                    "rouge1_f1": 0.0,
                    "rouge2_f1": 0.0,
                    "rougeL_f1": 0.0,
                    "bleu": 0.0
                },
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Generation truncated before completing text"
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
                metric_values={"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0, "bleu": 0.0},
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="Empty generation"
            )

        pred_tokens = re.findall(r"\w+", text.lower())
        ref_tokens = re.findall(r"\w+", ref_str.lower())

        r1_p, r1_r, r1_f1 = compute_rouge_n(pred_tokens, ref_tokens, n=1)
        r2_p, r2_r, r2_f1 = compute_rouge_n(pred_tokens, ref_tokens, n=2)
        rl_p, rl_r, rl_f1 = compute_rouge_l(pred_tokens, ref_tokens)
        bleu = compute_sentence_bleu(pred_tokens, ref_tokens)

        # Repetition metrics
        dist1 = len(set(pred_tokens)) / max(len(pred_tokens), 1)
        bigrams = _get_ngrams(pred_tokens, 2)
        dist2 = len(set(bigrams)) / max(len(bigrams), 1)

        # Consider generation adequate if ROUGE-L F1 >= 0.25 (or exact match)
        is_adequate = (rl_f1 >= 0.25) or (text.strip().lower() == ref_str.strip().lower())
        status = EvaluationStatus.COMPLETED_CORRECT if is_adequate else EvaluationStatus.COMPLETED_INCORRECT

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_str,
            raw_response=raw_output,
            normalized_response=text.strip(),
            parsed_answer=text.strip(),
            evaluation_status=status,
            answer_correct=is_adequate,
            metric_values={
                "rouge1_f1": round(r1_f1, 4),
                "rouge2_f1": round(r2_f1, 4),
                "rougeL_f1": round(rl_f1, 4),
                "bleu": round(bleu, 4),
                "distinct_1": round(dist1, 4),
                "distinct_2": round(dist2, 4),
                "pred_word_count": len(pred_tokens),
                "ref_word_count": len(ref_tokens)
            },
            parse_success=True,
            generation_truncated=False
        )
