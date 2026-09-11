"""Unit tests for GSM8K dataset loading, normalization, and leakage prevention."""

import unittest
import os
from src.data.gsm8k import load_gsm8k, parse_gsm8k_line, normalize_gold_answer


class TestGSM8KDataset(unittest.TestCase):

    def test_normalize_gold_answer(self):
        self.assertEqual(normalize_gold_answer("The answer is 42\n#### 42"), "42")
        self.assertEqual(normalize_gold_answer("#### $1,000"), "1000")
        self.assertEqual(normalize_gold_answer("#### -15"), "-15")
        self.assertEqual(normalize_gold_answer("#### 3.5"), "3.5")

    def test_parse_gsm8k_line(self):
        line = '{"question": "What is 2 + 2?", "answer": "2 plus 2 is 4.\\n#### 4"}'
        rec = parse_gsm8k_line(line, split="test", index=0)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.id, "gsm8k_test_0000")
        self.assertEqual(rec.question, "What is 2 + 2?")
        self.assertEqual(rec.answer, "4")
        self.assertEqual(rec.solution, "2 plus 2 is 4.")

    def test_dataset_train_test_separation(self):
        """Ensures test.jsonl and train.jsonl are distinct and non-empty."""
        test_records = load_gsm8k(split="test", eval_size=10)
        train_records = load_gsm8k(split="train", eval_size=10)

        self.assertEqual(len(test_records), 10)
        self.assertEqual(len(train_records), 10)

        test_ids = {r.id for r in test_records}
        train_ids = {r.id for r in train_records}
        self.assertTrue(test_ids.isdisjoint(train_ids))

        # Check that test questions are not identical to train questions in top samples
        test_q = {r.question for r in test_records}
        train_q = {r.question for r in train_records}
        self.assertTrue(test_q.isdisjoint(train_q))


if __name__ == "__main__":
    unittest.main()
