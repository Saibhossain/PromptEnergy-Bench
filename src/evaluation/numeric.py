"""Numeric and Mathematical Reasoning Evaluator for PromptEnergy-Bench.

Supports:
- Integers, floats, fractions (e.g. 3/4), scientific notation (1.2e-3), currencies, percentages.
- Standard answer patterns (#### [num], \\boxed{[num]}, explicit answer phrases, last standalone number).
- Relative and absolute tolerance numeric comparison.
- Strict rejection of truncated generations.
"""

import math
import re
from typing import Optional, Any, Dict

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)


class NumericEvaluator(BaseEvaluator):
    """Evaluates mathematical and numeric reasoning tasks."""

    def validate_config(self) -> None:
        if self.config.numeric_tolerance < 0:
            self.config.numeric_tolerance = 1e-5

    @staticmethod
    def normalize_number_string(val_str: Any) -> Optional[float]:
        """Converts raw string or numeric representation to a clean float."""
        if val_str is None:
            return None
        if isinstance(val_str, (int, float)):
            if math.isnan(val_str) or math.isinf(val_str):
                return None
            return float(val_str)

        cleaned = str(val_str).strip()
        # Clean currency symbols, commas, percent signs, and trailing punctuation
        cleaned = re.sub(r"[,\$€£%]", "", cleaned).strip()
        cleaned = cleaned.rstrip(".:;")

        # Handle simple fractions like "3/4" or "-1/2"
        if "/" in cleaned and len(cleaned.split("/")) == 2:
            parts = cleaned.split("/")
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass

        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def extract_answer(
        cls,
        raw_output: str,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False
    ) -> Optional[str]:
        """Extracts the predicted final numeric answer from model generation.
        
        Strictly returns None if generation_truncated is True.
        """
        if generation_truncated:
            return None
        if not raw_output or not str(raw_output).strip():
            return None

        # 1. Explicit '#### [number]' marker (strictly highest priority)
        if "####" in raw_output:
            after_hash = raw_output.split("####")[-1].strip()
            match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?", after_hash)
            if match:
                return re.sub(r"[,\$€£%]", "", match.group(0))

        # 2. LaTeX \boxed{...} pattern
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", raw_output)
        if boxed_match:
            cand = boxed_match.group(1).strip()
            match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?", re.sub(r"[,\$€£%]", "", cand))
            if match:
                return match.group(0)

        # Separate non-thinking visible text
        non_thinking = cls.strip_thinking(raw_output, raw_response)

        # 3. Explicit answer phrases in visible text first, then raw_output
        search_targets = [non_thinking, raw_output] if non_thinking else [raw_output]
        phrase_patterns = [
            r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|answer\s*[:=])\s*([\$€£]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?)",
            r"(?:equals?|total\s+is|result\s*[:=])\s*([\$€£]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?)"
        ]
        for target in search_targets:
            if not target:
                continue
            for pattern in phrase_patterns:
                matches = list(re.finditer(pattern, target, re.IGNORECASE))
                if matches:
                    candidate = matches[-1].group(1)
                    num_match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?", re.sub(r"[,\$€£%]", "", candidate))
                    if num_match:
                        return num_match.group(0)

        # 4. Fallback: find the last standalone number in visible text, or raw output
        fallback_target = non_thinking if non_thinking else raw_output
        numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?:[eE][-+]?\d+)?", fallback_target)
        if numbers:
            last_num = numbers[-1]
            return re.sub(r"[,\$€£%]", "", last_num)

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
        """Evaluates mathematical/numeric response."""
        ref_str = str(gold_answer) if gold_answer is not None else ""
        gold_norm = self.normalize_number_string(gold_answer)

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
                metric_values={"exact_match": False, "numeric_error": None},
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Generation stopped before completion (budget exceeded)"
            )

        if not raw_output or not str(raw_output).strip():
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response="",
                parsed_answer=None,
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"exact_match": False, "numeric_error": None},
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="Model produced empty output"
            )

        extracted = self.extract_answer(raw_output, raw_response, generation_truncated=False)
        if extracted is None:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response=self.strip_thinking(raw_output, raw_response),
                parsed_answer=None,
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"exact_match": False, "numeric_error": None},
                parse_success=False,
                generation_truncated=False,
                error_type="parse_failure",
                error_message="No numeric candidate found in output"
            )

        pred_norm = self.normalize_number_string(extracted)
        if pred_norm is None:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response=extracted,
                parsed_answer=extracted,
                evaluation_status=EvaluationStatus.INVALID_FORMAT,
                answer_correct=False,
                metric_values={"exact_match": False, "numeric_error": None},
                parse_success=False,
                generation_truncated=False,
                error_type="invalid_format",
                error_message="Extracted token could not be parsed as a valid number"
            )

        # Tolerance comparison
        tol = self.config.numeric_tolerance
        if gold_norm is not None:
            diff = abs(pred_norm - gold_norm)
            is_correct = math.isclose(pred_norm, gold_norm, rel_tol=tol, abs_tol=tol)
            exact_match = (pred_norm == gold_norm) or (extracted.strip() == ref_str.strip())
        else:
            diff = None
            is_correct = (extracted.strip() == ref_str.strip())
            exact_match = is_correct

        status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.COMPLETED_INCORRECT
        err_type = "none" if is_correct else "arithmetic_error"
        err_msg = None if is_correct else f"Predicted {pred_norm} does not match expected {gold_norm}"

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_str,
            raw_response=raw_output,
            normalized_response=str(pred_norm),
            parsed_answer=extracted,
            evaluation_status=status,
            answer_correct=is_correct,
            metric_values={
                "exact_match": exact_match,
                "numeric_error": diff,
                "relative_error": (diff / abs(gold_norm)) if (gold_norm and diff is not None and abs(gold_norm) > 1e-12) else 0.0
            },
            parse_success=True,
            generation_truncated=False,
            error_type=err_type,
            error_message=err_msg
        )
