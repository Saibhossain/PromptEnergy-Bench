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

    def test_incorrect_answer(self):
        output = "16 - 3 = 13. #### 13"
        res = GSM8KEvaluator.evaluate(output, "18")
        self.assertTrue(res.answer_parse_success)
        self.assertFalse(res.answer_correct)

    def test_empty_or_unparseable_output(self):
        res = GSM8KEvaluator.evaluate("", "18")
        self.assertFalse(res.answer_parse_success)
        self.assertFalse(res.answer_correct)
        self.assertIsNone(res.extracted_answer)


if __name__ == "__main__":
    unittest.main()
