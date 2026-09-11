"""Unit tests for GSM8K answer extraction and evaluation."""

import unittest
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator


class TestGSM8KEvaluator(unittest.TestCase):

    def test_explicit_hash_marker(self):
        output = "To find the eggs, we calculate 16 - 3 - 4 = 9.\n9 * 2 = 18.\n#### 18"
        res = GSM8KEvaluator.evaluate(output, "18")
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "18")

    def test_currency_and_comma_formatting(self):
        output = "The total value increased.\n#### $70,000"
        res = GSM8KEvaluator.evaluate(output, "70000")
        self.assertTrue(res.answer_correct)

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
        # Example from prompt: "The building has 120 units... 30 are unoccupied. #### 30"
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

    def test_truncation_rejection_without_explicit_answer(self):
        truncated_output = "<think>\nThe building has 120 units and 30 are"
        res = GSM8KEvaluator.evaluate(truncated_output, "30", generation_truncated=True)
        self.assertFalse(res.answer_parse_success)
        self.assertFalse(res.answer_correct)
        self.assertIsNone(res.extracted_answer)

    def test_truncation_with_explicit_answer_present(self):
        output = "Step 1: 15+15=30. #### 30"
        res = GSM8KEvaluator.evaluate(output, "30", generation_truncated=True)
        self.assertTrue(res.answer_parse_success)
        self.assertTrue(res.answer_correct)
        self.assertEqual(res.extracted_answer, "30")


if __name__ == "__main__":
    unittest.main()
