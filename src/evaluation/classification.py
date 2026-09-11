"""Classification Evaluator for PromptEnergy-Bench.

Supports:
- Binary and multiclass classification tasks (SST-2, AG News, BoolQ, etc.).
- Label extraction from textual outputs, synonyms mapping, and category resolution.
- Per-sample evaluation and batch metric calculations (Macro/Micro/Weighted F1, Precision, Recall, Confusion Matrix).
- Strict rejection of truncated generations.
"""

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


class ClassificationEvaluator(BaseEvaluator):
    """Evaluates classification tasks."""

    def validate_config(self) -> None:
        self.labels = [str(l).strip().lower() for l in (self.config.custom_labels or ["0", "1"])]
        # Map common synonyms to standard canonical labels
        self.synonyms: Dict[str, str] = {
            "positive": "positive", "pos": "positive", "good": "positive", "1": "1", "true": "true", "yes": "yes",
            "negative": "negative", "neg": "negative", "bad": "negative", "0": "0", "false": "false", "no": "no"
        }
        if self.config.extra_params.get("synonym_map"):
            self.synonyms.update({str(k).lower(): str(v).lower() for k, v in self.config.extra_params["synonym_map"].items()})

    def normalize_label(self, label: Any) -> str:
        """Normalizes raw label string."""
        s = str(label).strip().lower()
        return self.synonyms.get(s, s)

    def extract_label(
        self,
        raw_output: str,
        raw_response: Optional[str] = None
    ) -> Optional[str]:
        """Extracts candidate classification label from output."""
        non_thinking = self.strip_thinking(raw_output, raw_response)
        target = non_thinking if non_thinking else raw_output
        target_lower = target.lower()

        # 1. Look for explicit pattern: "Label: [class]", "Class: [class]", "Category: [class]"
        label_pattern = r"(?:label|class|category|classification|sentiment|prediction)\s*[:=]\s*([a-zA-Z0-9_\-]+)"
        match = re.search(label_pattern, target_lower)
        if match:
            cand = self.normalize_label(match.group(1))
            if cand in self.labels or cand in self.synonyms.values():
                return cand

        # 2. Check for explicit exact occurrences of configured labels
        found = []
        for lbl in self.labels:
            # Word boundary search for label
            if re.search(rf"\b{re.escape(lbl)}\b", target_lower):
                found.append(lbl)

        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            # Find the last mentioned valid label
            positions = [(target_lower.rfind(lbl), lbl) for lbl in found]
            positions.sort(key=lambda x: x[0])
            return positions[-1][1]

        # 3. Check for any synonym keywords
        for syn, canonical in self.synonyms.items():
            if re.search(rf"\b{re.escape(syn)}\b", target_lower):
                if canonical in self.labels:
                    return canonical

        return None

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates single classification instance."""
        ref_norm = self.normalize_label(gold_answer)

        if generation_truncated:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_norm,
                raw_response=raw_output,
                normalized_response=None,
                parsed_answer=None,
                evaluation_status=EvaluationStatus.GENERATION_TRUNCATED,
                answer_correct=False,
                metric_values={"exact_match": False},
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Generation truncated before classification decision"
            )

        if not raw_output or not str(raw_output).strip():
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_norm,
                raw_response=raw_output,
                normalized_response="",
                parsed_answer=None,
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"exact_match": False},
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="Empty model response"
            )

        extracted = self.extract_label(raw_output, raw_response)
        if extracted is None:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_norm,
                raw_response=raw_output,
                normalized_response=self.strip_thinking(raw_output, raw_response),
                parsed_answer=None,
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"exact_match": False},
                parse_success=False,
                generation_truncated=False,
                error_type="parse_failure",
                error_message="No recognizable class label found in response"
            )

        is_correct = (extracted == ref_norm)
        status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.COMPLETED_INCORRECT

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_norm,
            raw_response=raw_output,
            normalized_response=extracted,
            parsed_answer=extracted,
            evaluation_status=status,
            answer_correct=is_correct,
            metric_values={"exact_match": is_correct, "predicted_class": extracted, "gold_class": ref_norm},
            parse_success=True,
            generation_truncated=False
        )
