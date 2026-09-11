"""Comprehensive Unit Tests for Universal Dataset-Agnostic Evaluators.

Tests all 8+ evaluators across all lifecycle states:
- Correct, Incorrect, Truncated, Empty, Malformed, and Ambiguous outputs.
- Multiple reference policies (max, all, first).
- Numeric tolerance, fractions, currency, scientific notation.
- Configuration validation and error handling.
"""

import unittest
from src.evaluation import (
    BaseEvaluator,
    TaskConfig,
    TaskType,
    EvaluationResult,
    EvaluationStatus,
    MetricValidityStatus,
    ConfigurationError,
    EvaluatorRegistry,
    get_evaluator,
    NumericEvaluator,
    MultipleChoiceEvaluator,
    ClassificationEvaluator,
    OpenQAEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    CodeEvaluator,
    SafetyEvaluator,
    GSM8KEvaluator
)


class TestUniversalEvaluators(unittest.TestCase):
    """Tests all modular task evaluators."""

    def test_configuration_validation(self):
        """Tests that invalid configurations fail fast with ConfigurationError."""
        # Invalid task type
        with self.assertRaises(ConfigurationError):
            TaskConfig(task_type="invalid_future_task")

        # Empty input field
        with self.assertRaises(ConfigurationError):
            TaskConfig(task_type=TaskType.MATH, input_field="")

        # Empty reference field
        with self.assertRaises(ConfigurationError):
            TaskConfig(task_type=TaskType.MATH, reference_field="")

        # Invalid multiple reference policy
        with self.assertRaises(ConfigurationError):
            TaskConfig(task_type=TaskType.OPEN_QA, multiple_reference_policy="invalid_policy")

        # Negative tolerance
        with self.assertRaises(ConfigurationError):
            TaskConfig(task_type=TaskType.MATH, numeric_tolerance=-0.5)

    def test_evaluator_registry(self):
        """Tests EvaluatorRegistry factory for known datasets and task types."""
        eval_gsm = get_evaluator("gsm8k")
        self.assertIsInstance(eval_gsm, NumericEvaluator)

        eval_mmlu = get_evaluator("mmlu")
        self.assertIsInstance(eval_mmlu, MultipleChoiceEvaluator)

        eval_sst = get_evaluator("sst2")
        self.assertIsInstance(eval_sst, ClassificationEvaluator)

        eval_nq = get_evaluator("nq")
        self.assertIsInstance(eval_nq, OpenQAEvaluator)

        eval_cnn = get_evaluator("cnn_dailymail")
        self.assertIsInstance(eval_cnn, GenerationEvaluator)

        eval_code = get_evaluator("humaneval")
        self.assertIsInstance(eval_code, CodeEvaluator)

        eval_safety = get_evaluator("do_not_answer")
        self.assertIsInstance(eval_safety, SafetyEvaluator)

    def test_numeric_evaluator(self):
        """Tests NumericEvaluator for math problems."""
        evaluator = NumericEvaluator(TaskConfig(task_type=TaskType.MATH, numeric_tolerance=1e-4))

        # 1. Correct standard #### format
        res = evaluator.evaluate("Step 1: 20 + 22 = 42\n#### 42", "42")
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.exact_match)
        self.assertEqual(res.parsed_answer, "42")
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 2. Correct LaTeX \boxed{} format
        res = evaluator.evaluate(r"The solution is \boxed{3.14159}", "3.14159")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 3. Currency and commas
        res = evaluator.evaluate("He earns $1,500.50 each month.", "1500.5")
        self.assertTrue(res.answer_correct)

        # 4. Fractions
        res = evaluator.evaluate("The probability is 3/4.", "0.75")
        self.assertTrue(res.answer_correct)

        # 5. Strict Truncation Rejection
        res = evaluator.evaluate("Step 1: We calculate 42 and the answer is 42", "42", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertFalse(res.parse_success)
        self.assertIsNone(res.parsed_answer)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

        # 6. Incorrect answer
        res = evaluator.evaluate("The answer is 99", "42")
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_INCORRECT)

        # 7. Empty output
        res = evaluator.evaluate("", "42")
        self.assertFalse(res.answer_correct)
        self.assertFalse(res.parse_success)
        self.assertEqual(res.evaluation_status, EvaluationStatus.PARSE_FAILURE)

    def test_multiple_choice_evaluator(self):
        """Tests MultipleChoiceEvaluator."""
        evaluator = MultipleChoiceEvaluator(TaskConfig(
            task_type=TaskType.MULTIPLE_CHOICE,
            custom_labels=["A", "B", "C", "D"]
        ))

        # 1. Correct single choice
        res = evaluator.evaluate("The correct answer is (B).", "B")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.parsed_answer, "B")
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 2. Correct format with ####
        res = evaluator.evaluate("#### C", "C")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.parsed_answer, "C")

        # 3. Ambiguous choice
        res = evaluator.evaluate("The answer is A. On second thought, the answer is B.", "B")
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.AMBIGUOUS_ANSWER)

        # 4. Truncated
        res = evaluator.evaluate("Looking at options A and B...", "A", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_classification_evaluator(self):
        """Tests ClassificationEvaluator."""
        evaluator = ClassificationEvaluator(TaskConfig(
            task_type=TaskType.CLASSIFICATION,
            custom_labels=["positive", "negative"]
        ))

        # 1. Correct sentiment
        res = evaluator.evaluate("The sentiment of the review is positive.", "positive")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.parsed_answer, "positive")
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 2. Synonym mapping ("pos" -> "positive")
        res = evaluator.evaluate("Classification: pos", "positive")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.parsed_answer, "positive")

        # 3. Truncated
        res = evaluator.evaluate("The sentiment is pos", "positive", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_open_qa_evaluator(self):
        """Tests OpenQAEvaluator with normalization and multiple references."""
        evaluator_max = OpenQAEvaluator(TaskConfig(
            task_type=TaskType.OPEN_QA,
            multiple_reference_policy="max"
        ))

        # 1. Single reference SQuAD normalization (removes "the", punctuation)
        res = evaluator_max.evaluate("The United States of America.", "United States of America")
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.metric_values["exact_match"])
        self.assertGreaterEqual(res.metric_values["token_f1"], 0.99)

        # 2. Multiple references (matches 2nd reference)
        res = evaluator_max.evaluate("George Washington", ["G. Washington", "George Washington", "President Washington"])
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.metric_values["exact_match"])

        # 3. Partial overlap (Token F1)
        res = evaluator_max.evaluate("President George Washington", "George Washington")
        self.assertTrue(res.answer_correct)  # F1 is 2*(2/3 * 2/2) / (2/3 + 1) = 0.8
        self.assertAlmostEqual(res.metric_values["token_f1"], 0.8, places=2)

        # 4. Truncated
        res = evaluator_max.evaluate("George Wash", "George Washington", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_generation_evaluator(self):
        """Tests GenerationEvaluator with ROUGE and BLEU."""
        evaluator = GenerationEvaluator(TaskConfig(task_type=TaskType.GENERATION))

        reference = "The quick brown fox jumps over the lazy dog."
        candidate = "A quick brown fox jumped over the lazy dog."

        res = evaluator.evaluate(candidate, reference)
        self.assertTrue(res.parse_success)
        self.assertGreater(res.metric_values["rouge1_f1"], 0.7)
        self.assertGreater(res.metric_values["rougeL_f1"], 0.7)
        self.assertGreater(res.metric_values["bleu"], 0.3)

        # Truncated
        res_trunc = evaluator.evaluate("The quick brown fox", reference, generation_truncated=True)
        self.assertFalse(res_trunc.answer_correct)
        self.assertEqual(res_trunc.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_rag_evaluator(self):
        """Tests RAGEvaluator with grounding and retrieval metrics."""
        evaluator = RAGEvaluator(TaskConfig(task_type=TaskType.RAG))

        context = {
            "retrieved_context": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
            "gold_doc_ids": ["doc_1", "doc_2"],
            "retrieved_doc_ids": ["doc_1", "doc_3"]
        }

        # Grounded answer
        res = evaluator.evaluate(
            "The Eiffel Tower was finished in 1889 in Paris.",
            "1889",
            context=context
        )
        self.assertTrue(res.parse_success)
        self.assertGreater(res.metric_values["groundedness_score"], 0.8)
        self.assertEqual(res.metric_values["retrieval_recall"], 0.5)
        self.assertEqual(res.metric_values["retrieval_precision"], 0.5)

        # Truncated RAG
        res_trunc = evaluator.evaluate("The Eiffel Tower", "1889", generation_truncated=True)
        self.assertFalse(res_trunc.answer_correct)
        self.assertEqual(res_trunc.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_code_evaluator(self):
        """Tests CodeEvaluator with AST syntax check and sandbox execution."""
        evaluator = CodeEvaluator(TaskConfig(task_type=TaskType.CODE))

        # 1. Correct code with unit test
        code_output = "```python\ndef add(a, b):\n    return a + b\n```"
        context = {"test_cases": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]}
        res = evaluator.evaluate(code_output, "def add(a, b): return a + b", context=context)
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.metric_values["syntax_valid"])
        self.assertEqual(res.metric_values["pass_rate"], 1.0)
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 2. Syntax error
        bad_syntax = "```python\ndef add(a, b\n    return a + b\n```"
        res = evaluator.evaluate(bad_syntax, "def add(a, b): return a + b", context=context)
        self.assertFalse(res.answer_correct)
        self.assertFalse(res.metric_values["syntax_valid"])
        self.assertEqual(res.evaluation_status, EvaluationStatus.INVALID_FORMAT)

        # 3. Failing unit test
        failing_code = "```python\ndef add(a, b):\n    return a - b\n```"
        res = evaluator.evaluate(failing_code, "def add(a, b): return a + b", context=context)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.EXECUTION_FAILURE)

        # 4. Truncated code
        res = evaluator.evaluate("def add(a, b):", "def add(a, b): return a + b", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)

    def test_safety_evaluator(self):
        """Tests SafetyEvaluator with harmful vs benign prompts."""
        evaluator = SafetyEvaluator(TaskConfig(task_type=TaskType.SAFETY))

        # 1. Harmful prompt -> Correct refusal
        res = evaluator.evaluate("I cannot fulfill this request as it violates safety guidelines.", "refusal")
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.metric_values["is_refusal"])
        self.assertTrue(res.metric_values["correct_refusal"])
        self.assertFalse(res.metric_values["over_refusal"])
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_CORRECT)

        # 2. Benign prompt -> Model refused (Over-refusal)
        res = evaluator.evaluate("I am unable to answer how to bake a cake.", "safe")
        self.assertFalse(res.answer_correct)
        self.assertTrue(res.metric_values["is_refusal"])
        self.assertTrue(res.metric_values["over_refusal"])
        self.assertFalse(res.metric_values["correct_refusal"])
        self.assertEqual(res.evaluation_status, EvaluationStatus.COMPLETED_INCORRECT)

        # 3. Truncated safety check
        res = evaluator.evaluate("I cannot", "refusal", generation_truncated=True)
        self.assertFalse(res.answer_correct)
        self.assertEqual(res.evaluation_status, EvaluationStatus.GENERATION_TRUNCATED)


if __name__ == "__main__":
    unittest.main()
