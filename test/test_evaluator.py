"""Unit tests for GSM8K answer extraction and evaluation.

Tests:
- Correct answers (explicit ####, LaTeX \boxed, phrases, fallback numbers, fractions, currencies)
- Incorrect answers
- Parsing failures (empty, invalid text)
- Truncated generations (strict rejection under all circumstances)
- Exact-match validation
"""

import unittest
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator, EvaluationResult


class TestGSM8KEvaluator(unittest.TestCase):

    def test_explicit_hash_marker_correct(self):
        output = "To find the eggs, we calculate 16 - 3 - 4 = 9.\n9 * 2 = 18.\n#### 18"
        res = GSM8KEvaluator.evaluate(output, "18")
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertTrue(res.exact_match)
        self.assertEqual(res.extracted_answer, "18")

    def test_explicit_hash_marker_incorrect(self):
        output = "Calculation gave: #### 42"
        res = GSM8KEvaluator.evaluate(output, "18")
        self.assertTrue(res.answer_parse_success)
        self.assertFalse(res.answer_correct)
        self.assertFalse(res.exact_match)
        self.assertEqual(res.extracted_answer, "42")

    def test_boxed_latex_marker(self):
        output = "The total area is \\boxed{144} square meters."
        res = GSM8KEvaluator.evaluate(output, "144")
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "144")

    def test_currency_and_comma_formatting(self):
        output = "The total value increased.\n#### $70,000"
        res = GSM8KEvaluator.evaluate(output, "70000")
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "70000")

    def test_phrase_extraction(self):
        output = "First step is adding 5 + 5. The final answer is 10."
        res = GSM8KEvaluator.evaluate(output, "10")
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "10")

    def test_fraction_and_decimal_normalization(self):
        res1 = GSM8KEvaluator.evaluate("#### 3.0", "3")
        self.assertTrue(res1.answer_correct)

        res2 = GSM8KEvaluator.evaluate("#### 1/2", "0.5")
        self.assertTrue(res2.answer_correct)

    def test_precedence_over_earlier_numbers(self):
        output = "The building has 120 units... 30 are unoccupied. #### 30"
        res = GSM8KEvaluator.evaluate(output, "30")
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "30")

    def test_negative_and_comma_formatting(self):
        res1 = GSM8KEvaluator.evaluate("Temperature dropped. #### -15", "-15")
        self.assertTrue(res1.answer_correct)
        self.assertEqual(res1.extracted_answer, "-15")

        res2 = GSM8KEvaluator.evaluate("Total count is #### 1,200", "1200")
        self.assertTrue(res2.answer_correct)
        self.assertEqual(res2.extracted_answer, "1200")

    def test_parsing_failure_empty_and_unparseable(self):
        res_empty = GSM8KEvaluator.evaluate("", "18")
        self.assertFalse(res_empty.answer_parse_success)
        self.assertFalse(res_empty.answer_correct)
        self.assertIsNone(res_empty.extracted_answer)

        res_unparseable = GSM8KEvaluator.evaluate("I do not know the answer to this question.", "18")
        self.assertFalse(res_unparseable.answer_parse_success)
        self.assertFalse(res_unparseable.answer_correct)
        self.assertIsNone(res_unparseable.extracted_answer)

    def test_strict_truncation_rejection(self):
        # Truncated in thinking trace
        truncated_thinking = "<think>\nJanet has 16 eggs. 16 - 7 = 9. 9 * 2 = 18. The answer is"
        res1 = GSM8KEvaluator.evaluate(truncated_thinking, "18", generation_truncated=True)
        self.assertFalse(res1.answer_parse_success)
        self.assertFalse(res1.answer_correct)
        self.assertFalse(res1.exact_match)
        self.assertIsNone(res1.extracted_answer)
        self.assertTrue(res1.generation_truncated)

        # Truncated even if partial #### was generated
        truncated_hash = "Calculation: #### 18"
        res2 = GSM8KEvaluator.evaluate(truncated_hash, "18", generation_truncated=True)
        self.assertFalse(res2.answer_parse_success)
        self.assertFalse(res2.answer_correct)
        self.assertIsNone(res2.extracted_answer)
        self.assertTrue(res2.generation_truncated)


if __name__ == "__main__":
    unittest.main()
