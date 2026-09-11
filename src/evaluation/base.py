"""Base abstractions, configuration schema, and result structures for PromptEnergy-Bench Evaluation.

Adheres strictly to the empirical research framework:
- Explicit separation of sample lifecycle states and evaluation statuses.
- Zero-data-coercion policy for unmeasured or uncomputable metrics.
- Strict invalidation of truncated generations (never counted as correct).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Dict, Any, List, Optional, Union, Tuple


class TaskType(str, Enum):
    """Supported dataset task families."""
    MATH = "math"
    MULTIPLE_CHOICE = "multiple_choice"
    CLASSIFICATION = "classification"
    OPEN_QA = "open_qa"
    GENERATION = "generation"
    RAG = "rag"
    CODE = "code"
    SAFETY = "safety"


class EvaluationStatus(str, Enum):
    """Explicit lifecycle status assigned to every evaluated sample."""
    COMPLETED_CORRECT = "completed_correct"
    COMPLETED_INCORRECT = "completed_incorrect"
    GENERATION_TRUNCATED = "generation_truncated"
    PARSE_FAILURE = "parse_failure"
    INVALID_FORMAT = "invalid_format"
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    EXECUTION_FAILURE = "execution_failure"
    EVALUATOR_ERROR = "evaluator_error"


class MetricValidityStatus(str, Enum):
    """Validity status for aggregated metrics and denominators."""
    COMPUTABLE = "computable"
    INSUFFICIENT_VALID_SAMPLES = "insufficient_valid_samples"
    NO_REFERENCE_ANSWERS = "no_reference_answers"
    UNSUPPORTED_TASK_TYPE = "unsupported_task_type"
    MISSING_METRIC_DEPENDENCY = "missing_metric_dependency"
    UNAVAILABLE_MEASUREMENT = "unavailable_measurement"


class ConfigurationError(Exception):
    """Raised when a task or dataset configuration is invalid or missing required fields."""
    pass


@dataclass
class TaskConfig:
    """Universal configuration for dataset evaluation."""
    task_type: Union[TaskType, str]
    input_field: str = "question"
    reference_field: str = "answer"
    optional_choices_field: Optional[str] = "choices"
    answer_format: str = "numeric"  # e.g., 'numeric', 'option_letter', 'text', 'code', 'category'
    normalization_rules: List[str] = field(default_factory=lambda: ["strip_whitespace"])
    metric_set: List[str] = field(default_factory=lambda: ["accuracy", "exact_match"])
    numeric_tolerance: float = 1e-5
    multiple_reference_policy: str = "max"  # 'max', 'all', 'first'
    semantic_metric_enabled: bool = False
    safety_evaluation_enabled: bool = False
    custom_labels: Optional[List[str]] = None
    pass_k_list: List[int] = field(default_factory=lambda: [1])
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.task_type, str):
            try:
                self.task_type = TaskType(self.task_type.lower())
            except ValueError:
                valid_types = [t.value for t in TaskType]
                raise ConfigurationError(
                    f"Invalid task_type '{self.task_type}'. Must be one of: {valid_types}"
                )

        # Validate required fields
        if not self.input_field or not str(self.input_field).strip():
            raise ConfigurationError("TaskConfig 'input_field' cannot be empty.")
        if not self.reference_field or not str(self.reference_field).strip():
            raise ConfigurationError("TaskConfig 'reference_field' cannot be empty.")
        if self.numeric_tolerance < 0:
            raise ConfigurationError("TaskConfig 'numeric_tolerance' must be non-negative.")
        if self.multiple_reference_policy not in ("max", "all", "first"):
            raise ConfigurationError(
                f"Invalid multiple_reference_policy '{self.multiple_reference_policy}'. Must be 'max', 'all', or 'first'."
            )


@dataclass
class EvaluationResult:
    """Standardized result returned for every evaluated sample."""
    sample_id: Any
    task_type: str
    reference_answer: Any
    raw_response: Optional[str]
    normalized_response: Optional[str]
    parsed_answer: Any
    evaluation_status: EvaluationStatus
    answer_correct: bool
    metric_values: Dict[str, Any] = field(default_factory=dict)
    parse_success: bool = False
    generation_truncated: bool = False
    actual_token_counts: Dict[str, Optional[int]] = field(default_factory=lambda: {
        "configured_max_tokens": None,
        "actual_output_tokens": None,
        "actual_thinking_tokens": None,
        "actual_visible_tokens": None,
        "total_generated_tokens": None
    })
    latency_metrics: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "ttft_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": None
    })
    energy_metrics: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "energy_total_j": None,
        "energy_prefill_j": None,
        "energy_decode_j": None,
        "energy_embedding_j": None,
        "energy_retrieval_j": None,
        "energy_overhead_j": None,
        "energy_net_j": None,
        "idle_power_w": None,
        "energy_status": "unavailable"
    })
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Backward compatibility properties
    @property
    def gold_answer(self) -> Any:
        return self.reference_answer

    @property
    def raw_output(self) -> str:
        return self.raw_response or ""

    @property
    def extracted_answer(self) -> Any:
        return self.parsed_answer

    @property
    def answer_parse_success(self) -> bool:
        return self.parse_success

    @property
    def exact_match(self) -> bool:
        return bool(self.metric_values.get("exact_match", False))

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to standardized dictionary for persistence."""
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "reference_answer": self.reference_answer,
            "raw_response": self.raw_response,
            "normalized_response": self.normalized_response,
            "parsed_answer": self.parsed_answer,
            "evaluation_status": self.evaluation_status.value if isinstance(self.evaluation_status, EvaluationStatus) else str(self.evaluation_status),
            "answer_correct": self.answer_correct,
            "metric_values": self.metric_values,
            "parse_success": self.parse_success,
            "generation_truncated": self.generation_truncated,
            "actual_token_counts": self.actual_token_counts,
            "latency_metrics": self.latency_metrics,
            "energy_metrics": self.energy_metrics,
            "error_type": self.error_type,
            "error_message": self.error_message,
            # Flat compatibility fields for legacy tables and visualizers
            "gold_answer": self.reference_answer,
            "raw_output": self.raw_response or "",
            "extracted_answer": self.parsed_answer,
            "answer_parse_success": self.parse_success,
            "exact_match": self.exact_match
        }


class BaseEvaluator(ABC):
    """Abstract base class for all task evaluators."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.validate_config()

    def validate_config(self) -> None:
        """Validates configuration parameters for this specific evaluator."""
        pass

    @abstractmethod
    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates a single sample with strict lifecycle guarantees.
        
        Must guarantee:
        - If generation_truncated is True, status is GENERATION_TRUNCATED, answer_correct is False,
          and parsed_answer is None.
        - Preserves both raw_response and normalized_response.
        - Populates task-appropriate metric_values.
        """
        pass

    @staticmethod
    def strip_thinking(raw_output: str, raw_response: Optional[str] = None) -> str:
        """Extracts visible non-thinking text from output containing <think>...</think> tags."""
        if raw_response and raw_response.strip():
            return raw_response.strip()
        if not raw_output:
            return ""
        non_thinking = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        if "<think>" in non_thinking:
            non_thinking = non_thinking.split("<think>")[0].strip()
        return non_thinking
