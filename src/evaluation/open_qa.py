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


def compute_span_match(prediction: str, ground_truth: str) -> bool:
    """Computes bidirectional span containment match for open QA (e.g. passage-based NQ references)."""
    norm_p = normalize_answer(prediction)
    norm_g = normalize_answer(ground_truth)
    if not norm_p or not norm_g:
        return False
    # Direct bidirectional substring containment
    if len(norm_p) >= 2 and norm_p in norm_g:
        return True
    if len(norm_g) >= 2 and norm_g in norm_p:
        return True

    # If prediction is a sentence ending with the answer entity (e.g., "...filmed in Vancouver")
    words_p = norm_p.split()
    if len(words_p) >= 2:
        stops = {"the", "a", "an", "is", "was", "are", "were", "in", "at", "on", "to", "for", "of", "and", "or"}
        # Check trailing spans from 1 up to 6 words
        for k in range(min(6, len(words_p)), 0, -1):
            sub_span = " ".join(words_p[-k:])
            if len(sub_span) >= 3 and sub_span not in stops and sub_span in norm_g:
                return True

    return False


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

        # 1. Search for explicit answer pattern (case-insensitive)
        # Matches inline answers and markdown headers like:
        # "**Final Answer:**\n<actual text>" or "**Answer:** <actual text>" or "Answer: <text>"
        pattern = r"(?:the\s+answer\s+is|final\s+answer\s*[:=]|answer\s*[:=]|conclusion\s*[:=])\s*(.*)"
        matches = list(re.finditer(pattern, target, re.IGNORECASE))
        if matches:
            phrase_match = matches[-1]
            rest_of_line = phrase_match.group(1).split("\n")[0].strip()
            # Strip markdown formatting markers like ** or ## or `
            cleaned_rest = re.sub(r"^[*_#`\s]+|[*_#`\s]+$", "", rest_of_line)
            if cleaned_rest:
                bold_in_line = re.search(r"\*\*([^*]+)\*\*", rest_of_line)
                if bold_in_line and 0 < len(bold_in_line.group(1).strip()) < 80:
                    return bold_in_line.group(1).strip()
                return cleaned_rest

            # If the rest of the line was empty or only markdown delimiters (e.g. "**Final Answer:**\n"),
            # grab the subsequent non-empty line
            post_match = target[phrase_match.end():]
            for raw_line in post_match.split("\n"):
                cleaned_line = re.sub(r"^[*_#`\s]+|[*_#`\s]+$", "", raw_line.strip())
                if cleaned_line:
                    bold_in_line = re.search(r"\*\*([^*]+)\*\*", raw_line)
                    if bold_in_line and 0 < len(bold_in_line.group(1).strip()) < 80:
                        return bold_in_line.group(1).strip()
                    return cleaned_line

        # 2. Check for markdown bold answer near the end, e.g. "**Vancouver**"
        bold_matches = re.findall(r"\*\*([^*]+)\*\*", target)
        if bold_matches:
            last_bold = bold_matches[-1].strip()
            if 0 < len(last_bold) < 120 and not last_bold.lower().startswith(("step", "note", "analysis", "conclusion")):
                # Check if this bold text occurs in the final segment of text
                last_segment = target[-300:] if len(target) > 300 else target
                if last_bold in last_segment:
                    return last_bold

        # 3. Fallback: return the last non-empty line of response (not lines[0], which is CoT intro)
        lines = [l.strip() for l in target.split("\n") if l.strip()]
        if lines:
            for line in reversed(lines):
                cleaned = re.sub(r"^[*_#`\s]+|[*_#`\s]+$", "", line)
                if cleaned and not cleaned.lower().startswith(("step", "note:", "source:")):
                    return cleaned
            return lines[-1]

        return target.strip()

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

        has_valid_reference = any(str(r).strip() for r in references)
        if not has_valid_reference:
            extracted = self.extract_answer(raw_output, raw_response)
            norm_pred = normalize_answer(extracted)
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_repr,
                raw_response=raw_output,
                normalized_response=norm_pred,
                parsed_answer=extracted,
                evaluation_status=EvaluationStatus.NO_REFERENCE,
                answer_correct=None,
                metric_values={"exact_match": None, "token_f1": None, "span_match": None, "has_reference": False},
                parse_success=True,
                generation_truncated=generation_truncated
            )

        extracted = self.extract_answer(raw_output, raw_response)
        norm_pred = normalize_answer(extracted)

        # Multi-reference scoring
        em_scores = [compute_exact_match(extracted, ref) for ref in references]
        f1_scores = [compute_f1(extracted, ref) for ref in references]
        span_scores = [compute_span_match(extracted, ref) for ref in references]

        policy = self.config.multiple_reference_policy
        if policy == "first":
            exact_match = em_scores[0]
            token_f1 = f1_scores[0]
            span_match = span_scores[0]
        elif policy == "all":
            exact_match = all(em_scores)
            token_f1 = sum(f1_scores) / len(f1_scores)
            span_match = all(span_scores)
        else:  # "max" default
            exact_match = any(em_scores)
            token_f1 = max(f1_scores)
            span_match = any(span_scores)

        # For Open-QA, consider answer correct if exact match, high F1 (>= 0.8), or span containment in reference
        is_correct = exact_match or (token_f1 >= 0.8) or span_match
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
                "span_match": span_match,
                "best_reference": references[f1_scores.index(max(f1_scores))] if references else ""
            },
            parse_success=True,
            generation_truncated=False
        )
