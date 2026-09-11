"""GSM8K Evaluator (Backward-compatible wrapper around universal NumericEvaluator).

Adheres strictly to the universal evaluation framework:
- Rejects truncated generations (answer_parse_success=False, answer_correct=False, parsed_answer=None).
- Multi-stage numeric extraction (####, \\boxed{}, phrases, standalone numbers).
- Exact numeric and tolerance comparison.
"""

from typing import Optional, Any, Dict

from src.evaluation.base import (
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)
from src.evaluation.numeric import NumericEvaluator


class GSM8KEvaluator(NumericEvaluator):
    """GSM8K mathematical evaluator implementing the universal evaluator interface."""

    def __init__(self, config: Optional[TaskConfig] = None):
        if config is None:
            config = TaskConfig(
                task_type=TaskType.MATH,
                input_field="question",
                reference_field="answer",
                answer_format="numeric",
                numeric_tolerance=1e-5
            )
        super().__init__(config)

    @classmethod
    def evaluate_sample(
        cls,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Class method convenience helper."""
        evaluator = cls()
        return evaluator.evaluate(
            raw_output=raw_output,
            gold_answer=gold_answer,
            raw_response=raw_response,
            generation_truncated=generation_truncated,
            sample_id=sample_id,
            context=context
        )

    # Legacy classmethod override for compatibility with old code calling GSM8KEvaluator.evaluate(...)
    @classmethod
    def evaluate(
        cls,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        evaluator = cls()
        return super(GSM8KEvaluator, evaluator).evaluate(
            raw_output=raw_output,
            gold_answer=gold_answer,
            raw_response=raw_response,
            generation_truncated=generation_truncated,
            sample_id=sample_id,
            context=context
        )
