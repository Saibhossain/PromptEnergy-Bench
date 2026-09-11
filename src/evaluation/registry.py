"""Evaluator Registry and Factory for PromptEnergy-Bench.

Enables dataset-agnostic evaluator instantiation via task types, dataset aliases, or custom TaskConfig.
"""

from typing import Dict, Any, Optional, Type, Union

from src.evaluation.base import (
    BaseEvaluator,
    TaskConfig,
    TaskType,
    ConfigurationError
)
from src.evaluation.numeric import NumericEvaluator
from src.evaluation.multiple_choice import MultipleChoiceEvaluator
from src.evaluation.classification import ClassificationEvaluator
from src.evaluation.open_qa import OpenQAEvaluator
from src.evaluation.generation import GenerationEvaluator
from src.evaluation.rag import RAGEvaluator
from src.evaluation.code import CodeEvaluator
from src.evaluation.safety import SafetyEvaluator


EVALUATOR_MAP: Dict[TaskType, Type[BaseEvaluator]] = {
    TaskType.MATH: NumericEvaluator,
    TaskType.MULTIPLE_CHOICE: MultipleChoiceEvaluator,
    TaskType.CLASSIFICATION: ClassificationEvaluator,
    TaskType.OPEN_QA: OpenQAEvaluator,
    TaskType.GENERATION: GenerationEvaluator,
    TaskType.RAG: RAGEvaluator,
    TaskType.CODE: CodeEvaluator,
    TaskType.SAFETY: SafetyEvaluator
}

DATASET_DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Math
    "gsm8k": {"task_type": TaskType.MATH, "answer_format": "numeric", "numeric_tolerance": 1e-5},
    "math": {"task_type": TaskType.MATH, "answer_format": "numeric", "numeric_tolerance": 1e-5},
    "svamp": {"task_type": TaskType.MATH, "answer_format": "numeric", "numeric_tolerance": 1e-5},
    # Multiple Choice
    "mmlu": {"task_type": TaskType.MULTIPLE_CHOICE, "answer_format": "option_letter", "custom_labels": ["A", "B", "C", "D"]},
    "arc": {"task_type": TaskType.MULTIPLE_CHOICE, "answer_format": "option_letter", "custom_labels": ["A", "B", "C", "D"]},
    "hellaswag": {"task_type": TaskType.MULTIPLE_CHOICE, "answer_format": "option_letter", "custom_labels": ["A", "B", "C", "D", "0", "1", "2", "3"]},
    # Classification
    "sst2": {"task_type": TaskType.CLASSIFICATION, "custom_labels": ["positive", "negative", "0", "1"]},
    "ag_news": {"task_type": TaskType.CLASSIFICATION, "custom_labels": ["world", "sports", "business", "sci/tech", "1", "2", "3", "4"]},
    "boolq": {"task_type": TaskType.CLASSIFICATION, "custom_labels": ["true", "false", "yes", "no"]},
    # Open QA
    "nq": {"task_type": TaskType.OPEN_QA, "answer_format": "text", "multiple_reference_policy": "max"},
    "squad": {"task_type": TaskType.OPEN_QA, "answer_format": "text", "multiple_reference_policy": "max"},
    "triviaqa": {"task_type": TaskType.OPEN_QA, "answer_format": "text", "multiple_reference_policy": "max"},
    # Generation / Summarization
    "cnn_dailymail": {"task_type": TaskType.GENERATION, "answer_format": "text"},
    "xsum": {"task_type": TaskType.GENERATION, "answer_format": "text"},
    # Code
    "humaneval": {"task_type": TaskType.CODE, "answer_format": "code"},
    "mbpp": {"task_type": TaskType.CODE, "answer_format": "code"},
    # Safety
    "do_not_answer": {"task_type": TaskType.SAFETY, "answer_format": "refusal"},
    "advglue": {"task_type": TaskType.SAFETY, "answer_format": "refusal"}
}


class EvaluatorRegistry:
    """Registry managing evaluator creation and dataset mappings."""

    @classmethod
    def register(cls, task_type: TaskType, evaluator_cls: Type[BaseEvaluator]) -> None:
        """Registers a new evaluator class for a task type."""
        EVALUATOR_MAP[task_type] = evaluator_cls

    @classmethod
    def get_evaluator(
        cls,
        task_type_or_dataset: Union[TaskType, str, TaskConfig],
        config: Optional[TaskConfig] = None,
        **kwargs: Any
    ) -> BaseEvaluator:
        """Instantiates an evaluator for a task type, dataset name, or TaskConfig."""
        if isinstance(task_type_or_dataset, TaskConfig):
            final_cfg = task_type_or_dataset
        elif config is not None:
            final_cfg = config
        else:
            # Check if dataset alias exists
            ds_key = str(task_type_or_dataset).lower()
            if ds_key in DATASET_DEFAULT_CONFIGS:
                cfg_params = dict(DATASET_DEFAULT_CONFIGS[ds_key])
                cfg_params.update(kwargs)
                final_cfg = TaskConfig(**cfg_params)
            else:
                # Treat as task_type
                try:
                    ttype = TaskType(ds_key)
                except ValueError:
                    valid_options = list(DATASET_DEFAULT_CONFIGS.keys()) + [t.value for t in TaskType]
                    raise ConfigurationError(
                        f"Unknown task type or dataset '{task_type_or_dataset}'. Supported options: {valid_options}"
                    )
                cfg_params = {"task_type": ttype}
                cfg_params.update(kwargs)
                final_cfg = TaskConfig(**cfg_params)

        evaluator_cls = EVALUATOR_MAP.get(final_cfg.task_type)
        if not evaluator_cls:
            raise ConfigurationError(f"No evaluator registered for task type '{final_cfg.task_type}'.")

        return evaluator_cls(final_cfg)


def get_evaluator(
    task_type_or_dataset: Union[TaskType, str, TaskConfig] = "gsm8k",
    config: Optional[TaskConfig] = None,
    **kwargs: Any
) -> BaseEvaluator:
    """Convenience factory function."""
    return EvaluatorRegistry.get_evaluator(task_type_or_dataset, config=config, **kwargs)
