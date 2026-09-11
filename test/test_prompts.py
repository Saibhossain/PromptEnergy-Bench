"""Unit tests for prompt formatting, 3-shot determinism, and hashing."""

import unittest
from src.prompts.gsm8k_prompts import (
    PromptStrategy,
    format_gsm8k_prompt,
    get_prompt_hash,
    FIXED_FEW_SHOT_EXAMPLES
)


class TestPrompts(unittest.TestCase):

    def test_all_strategies_format(self):
        question = "Janet has 16 eggs. She eats 3. How many left?"
        for strat in PromptStrategy:
            msgs = format_gsm8k_prompt(strat, question)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertIn("#### [number]", msgs[0]["content"] + msgs[1]["content"])

    def test_few_shot_3_determinism(self):
        """Ensures 3-shot examples remain fixed and deterministic."""
        self.assertEqual(len(FIXED_FEW_SHOT_EXAMPLES), 3)
        self.assertIn("Natalia sold clips", FIXED_FEW_SHOT_EXAMPLES[0]["problem"])
        self.assertIn("Weng earns", FIXED_FEW_SHOT_EXAMPLES[1]["problem"])
        self.assertIn("Betty is saving money", FIXED_FEW_SHOT_EXAMPLES[2]["problem"])

        # Format across two distinct questions
        msgs_a = format_gsm8k_prompt(PromptStrategy.FEW_SHOT_3, "Question A")
        msgs_b = format_gsm8k_prompt(PromptStrategy.FEW_SHOT_3, "Question B")

        # Few-shot examples section must be identical
        user_a = msgs_a[1]["content"]
        user_b = msgs_b[1]["content"]
        self.assertTrue(user_a.startswith("Example 1:\nProblem: Natalia"))
        self.assertTrue(user_b.startswith("Example 1:\nProblem: Natalia"))

    def test_prompt_hash_reproducibility(self):
        h1 = get_prompt_hash("v1.0")
        h2 = get_prompt_hash("v1.0")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)


if __name__ == "__main__":
    unittest.main()
