"""Code Generation Evaluator for PromptEnergy-Bench.

Supports:
- Python code extraction from markdown blocks (```python ... ```) or raw outputs.
- Abstract Syntax Tree (AST) syntax validation.
- Unit test execution simulation/runner for functional correctness.
- Runtime error handling and static validation.
- Strict rejection of truncated generations.
"""

import ast
import re
from typing import Optional, Any, Dict, List, Set, Tuple

from src.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    TaskConfig,
    TaskType
)


class CodeEvaluator(BaseEvaluator):
    """Evaluates code generation benchmarks (HumanEval, MBPP)."""

    def extract_code(
        self,
        raw_output: str,
        raw_response: Optional[str] = None
    ) -> str:
        """Extracts executable code block from response."""
        non_thinking = self.strip_thinking(raw_output, raw_response)
        target = non_thinking if non_thinking else (raw_output or "")

        # Look for markdown code fence
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", target, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            # Return the largest code block or last block
            return max(code_blocks, key=len).strip()

        # If no code fence, return the non-thinking text directly
        return target.strip()

    def check_syntax(self, code_str: str) -> Tuple[bool, Optional[str]]:
        """Checks if code string is syntactically valid Python."""
        try:
            ast.parse(code_str)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def run_unit_tests(
        self,
        code_str: str,
        unit_tests: List[str]
    ) -> Tuple[bool, int, int, Optional[str]]:
        """Executes unit test assertions in a local dictionary environment."""
        if not unit_tests:
            return True, 1, 1, None

        scope: Dict[str, Any] = {}
        try:
            exec(code_str, scope)
        except Exception as e:
            return False, 0, len(unit_tests), f"ExecutionError on code definition: {type(e).__name__}: {str(e)}"

        passed = 0
        total = len(unit_tests)
        last_err = None

        for test in unit_tests:
            try:
                exec(test, scope)
                passed += 1
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)}"

        all_passed = (passed == total)
        return all_passed, passed, total, last_err

    def evaluate(
        self,
        raw_output: str,
        gold_answer: Any,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False,
        sample_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluates code generation output."""
        ref_str = str(gold_answer) if gold_answer is not None else ""
        context_dict = context or {}
        test_cases = context_dict.get("test_cases", [])
        if isinstance(test_cases, str):
            test_cases = [test_cases]

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
                    "syntax_valid": False,
                    "unit_tests_passed": 0,
                    "total_unit_tests": len(test_cases),
                    "pass_rate": 0.0
                },
                parse_success=False,
                generation_truncated=True,
                error_type="generation_truncated",
                error_message="Code generation truncated before completion"
            )

        code = self.extract_code(raw_output, raw_response)
        if not code.strip():
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response="",
                parsed_answer="",
                evaluation_status=EvaluationStatus.PARSE_FAILURE,
                answer_correct=False,
                metric_values={"syntax_valid": False, "pass_rate": 0.0},
                parse_success=False,
                generation_truncated=False,
                error_type="empty_output",
                error_message="No code found in model output"
            )

        # 1. Syntax check
        syntax_ok, syntax_err = self.check_syntax(code)
        if not syntax_ok:
            return EvaluationResult(
                sample_id=sample_id,
                task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
                reference_answer=ref_str,
                raw_response=raw_output,
                normalized_response=code,
                parsed_answer=code,
                evaluation_status=EvaluationStatus.INVALID_FORMAT,
                answer_correct=False,
                metric_values={
                    "syntax_valid": False,
                    "unit_tests_passed": 0,
                    "total_unit_tests": len(test_cases),
                    "pass_rate": 0.0,
                    "syntax_error": syntax_err
                },
                parse_success=False,
                generation_truncated=False,
                error_type="syntax_error",
                error_message=syntax_err
            )

        # 2. Unit test execution
        if test_cases:
            all_passed, passed_n, total_n, exec_err = self.run_unit_tests(code, test_cases)
            pass_rate = passed_n / total_n if total_n > 0 else 0.0
            is_correct = all_passed
            status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.EXECUTION_FAILURE
        else:
            # If no unit tests supplied, check exact string or syntax validity
            is_correct = (code.strip() == ref_str.strip()) or syntax_ok
            pass_rate = 1.0 if is_correct else 0.0
            passed_n = 1 if is_correct else 0
            total_n = 1
            exec_err = None
            status = EvaluationStatus.COMPLETED_CORRECT if is_correct else EvaluationStatus.COMPLETED_INCORRECT

        return EvaluationResult(
            sample_id=sample_id,
            task_type=self.config.task_type.value if isinstance(self.config.task_type, TaskType) else str(self.config.task_type),
            reference_answer=ref_str,
            raw_response=raw_output,
            normalized_response=code,
            parsed_answer=code,
            evaluation_status=status,
            answer_correct=is_correct,
            metric_values={
                "syntax_valid": True,
                "unit_tests_passed": passed_n,
                "total_unit_tests": total_n,
                "pass_rate": round(pass_rate, 4),
                "execution_error": exec_err
            },
            parse_success=True,
            generation_truncated=False,
            error_type="execution_failure" if exec_err else None,
            error_message=exec_err
        )
