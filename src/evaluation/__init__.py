"""Universal evaluation and metric computation package for PromptEnergy-Bench."""

from src.evaluation.base import (
    BaseEvaluator,
    TaskConfig,
    TaskType,
    EvaluationResult,
    EvaluationStatus,
    MetricValidityStatus,
    ConfigurationError
)
from src.evaluation.registry import EvaluatorRegistry, get_evaluator
from src.evaluation.numeric import NumericEvaluator
from src.evaluation.multiple_choice import MultipleChoiceEvaluator
from src.evaluation.classification import ClassificationEvaluator
from src.evaluation.open_qa import OpenQAEvaluator
from src.evaluation.generation import GenerationEvaluator
from src.evaluation.rag import RAGEvaluator
from src.evaluation.code import CodeEvaluator
from src.evaluation.safety import SafetyEvaluator
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.evaluation.metrics import (
    compute_experiment_metrics,
    compute_strategy_summary,
    calculate_meg,
    calculate_mag_token,
    calculate_mag_latency,
    solve_budget_constrained_prompting
)

__all__ = [
    "BaseEvaluator",
    "TaskConfig",
    "TaskType",
    "EvaluationResult",
    "EvaluationStatus",
    "MetricValidityStatus",
    "ConfigurationError",
    "EvaluatorRegistry",
    "get_evaluator",
    "NumericEvaluator",
    "MultipleChoiceEvaluator",
    "ClassificationEvaluator",
    "OpenQAEvaluator",
    "GenerationEvaluator",
    "RAGEvaluator",
    "CodeEvaluator",
    "SafetyEvaluator",
    "GSM8KEvaluator",
    "compute_experiment_metrics",
    "compute_strategy_summary",
    "calculate_meg",
    "calculate_mag_token",
    "calculate_mag_latency",
    "solve_budget_constrained_prompting"
]
