"""Multiple Choice Evaluator for PromptEnergy-Bench.

Supports:
- Normalized option letter/number extraction (A/B/C/D, 1/2/3/4, or arbitrary choice labels).
- Standard markers (e.g. 'The answer is (A)', 'Option B', 'Answer: C', '#### D', '\\boxed{A}').
- Ambiguity detection (flags when model picks multiple contradictory options).
- Invalid option detection (choices outside allowed candidate set).
- Strict rejection of truncated generations.
"""

import re
from typing import Optional, Any, Dict, List, Set, Tuple

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)


class MultipleChoiceEvaluator(BaseEvaluator):
    """Evaluates multiple-choice reasoning datasets (MMLU, ARC, CommonsenseQA, etc.)."""

    def validate_config(self) -> None:
        if not self.config.custom_labels:
            self.allowed_options = ["A", "B", "C", "D", "E", "F", "G", "H", "1", "2", "3", "4"]
        else:
            self.allowed_options = [str(lbl).strip().upper() for lbl in self.config.custom_labels]

    def extract_choice(
        self,
        raw_output: str,
        raw_response: Optional[str] = None,
        allowed_options: Optional[List[str]] = None
    ) -> Tuple[Optional[str], bool]:
        """Extracts the selected choice option.
        
        Returns:
            (extracted_option, is_ambiguous)
        """
        valid_set = set(allowed_options or self.allowed_options)
        non_thinking = self.strip_thinking(raw_output, raw_response)
        target = non_thinking if non_thinking else raw_output

        # 1. Check for explicit '#### [Option]' or LaTeX \boxed{[Option]}
        hash_match = re.search(r"####\s*\(?([A-Za-z0-9])\)?", raw_output)
        if hash_match:
            cand = hash_match.group(1).upper()
            if cand in valid_set:
                return cand, False

        boxed_match = re.search(r"\\boxed\{\s*\(?([A-Za-z0-9])\)?\s*\}", raw_output)
        if boxed_match:
            cand = boxed_match.group(1).upper()
            if cand in valid_set:
                return cand, False

        # 2. Check for explicit phrase patterns: 'The answer is (A)', 'Answer: B', 'Choice: C', 'Option: D'
        phrase_patterns = [
            r"(?:the\s+correct\s+answer\s+is|the\s+answer\s+is|correct\s+option\s+is|answer\s*[:=]|option\s*[:=]|choice\s*[:=])\s*\(?([A-Za-z0-9])\)?",
            r"(?:choose|select)\s*\(?([A-Za-z0-9])\)?",
            r"\(([A-Za-z0-9])\)\s*(?:is\s+correct|is\s+the\s+answer)"
        ]

        found_cands = []
        for pat in phrase_patterns:
            for match in re.finditer(pat, target, re.IGNORECASE):
                c = match.group(1).upper()
                if c in valid_set:
                    found_cands.append(c)

        if found_cands:
            unique_cands = list(dict.fromkeys(found_cands))
            if len(unique_cands) == 1:
                return unique_cands[0], False
            elif len(unique_cands) > 1:
                # Ambiguous if multiple distinct choices were stated as the answer
                return unique_cands[-1], True

        # 3. Check for trailing standalone choice marker e.g., "\n(A)" or "\nAnswer: A"
        trailing_matches = re.findall(r"(?:^|\s|\()([A-Za-z0-9])(?:\)|\.|\s|$)", target)
        valid_trailing = [m.upper() for m in trailing_matches if m.upper() in valid_set]
        if valid_trailing:
            # Check if multiple contradictory options appear in close proximity
            if len(set(valid_trailing[-2:])) > 1 and len(valid_trailing) > 2:
                return valid_trailing[-1], True
            return valid_trailing[-1], False

        return None, False

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates multiple-choice model output."""
        ref_norm = str(gold_answer).strip().upper() if gold_answer is not None else ""
        # Handle formats like '(A)' or 'A.' in reference
        clean_ref_match = re.search(r"([A-Za-z0-9])", ref_norm)
        if clean_ref_match:
            ref_norm = clean_ref_match.group(1).upper()

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
                error_message="Generation truncated before option selection"
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

        extracted, is_ambiguous = self.extract_choice(raw_output, raw_response)

        if is_ambiguous:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_norm,
                raw_response=raw_output,
                normalized_response=extracted,
                parsed_answer=extracted,
                evaluation_status=EvaluationStatus.AMBIGUOUS_ANSWER,
                answer_correct=False,
                metric_values={"exact_match": False, "ambiguous": True},
                parse_success=False,
                generation_truncated=False,
                error_type="ambiguous_answer",
                error_message="Model proposed multiple contradictory choice options"
            )

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
                error_message="No valid multiple-choice option found in response"
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
            metric_values={"exact_match": is_correct, "ambiguous": False},
            parse_success=True,
            generation_truncated=False
        )
