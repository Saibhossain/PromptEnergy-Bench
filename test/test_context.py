"""Unit tests for Controlled Context Scaling Builder."""

import unittest
from src.data.gsm8k import GSM8KRecord
from src.data.context_builder import ContextBuilder, ScaledContext, estimate_tokens


class TestContextBuilder(unittest.TestCase):

    def setUp(self):
        self.mock_train = [
            GSM8KRecord(
                id="gsm8k_train_0001",
                question="John buys 5 apples for $2 each. How much does he pay?",
                answer="10",
                solution="5 * 2 = 10",
                raw_answer="5 * 2 = 10\n#### 10"
            ),
            GSM8KRecord(
                id="gsm8k_train_0002",
                question="Mary has 12 oranges. She bakes cakes using 4 oranges each. How many cakes can she bake?",
                answer="3",
                solution="12 / 4 = 3",
                raw_answer="12 / 4 = 3\n#### 3"
            ),
            GSM8KRecord(
                id="gsm8k_train_0003",
                question="An airplane travels 500 miles per hour for 3 hours. How far does it travel?",
                answer="1500",
                solution="500 * 3 = 1500",
                raw_answer="500 * 3 = 1500\n#### 1500"
            )
        ]
        self.builder = ContextBuilder(train_records=self.mock_train)
        self.target = GSM8KRecord(
            id="gsm8k_test_0001",
            question="Alice buys 3 apples at $2 each. What is her total?",
            answer="6",
            solution="3 * 2 = 6",
            raw_answer="3 * 2 = 6\n#### 6"
        )

    def test_relevant_context_prioritizes_apple_problem(self):
        ctx: ScaledContext = self.builder.build_context(
            target_record=self.target,
            context_type="relevant",
            target_tokens=100
        )
        self.assertEqual(ctx.context_type, "relevant")
        self.assertGreater(ctx.actual_context_tokens, 0)
        self.assertIn("gsm8k_train_0001", ctx.context_document_ids[0])

    def test_distractor_context_deprioritizes_apple_problem(self):
        ctx: ScaledContext = self.builder.build_context(
            target_record=self.target,
            context_type="distractor",
            target_tokens=100
        )
        self.assertEqual(ctx.context_type, "distractor")
        # Distractor should choose airplane problem before apple problem
        self.assertIn("gsm8k_train_0003", ctx.context_document_ids[0])

    def test_context_scaling_tokens(self):
        ctx_50: ScaledContext = self.builder.build_context(self.target, "relevant", target_tokens=30)
        ctx_200: ScaledContext = self.builder.build_context(self.target, "relevant", target_tokens=200)
        self.assertLessEqual(ctx_50.actual_context_tokens, ctx_200.actual_context_tokens)


if __name__ == "__main__":
    unittest.main()
