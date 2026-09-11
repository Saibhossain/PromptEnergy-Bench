"""Comprehensive Unit Tests for Metrics Computation.

Covers:
- Correct answers and accuracy computation
- Incorrect answers
- Parsing failures
- Truncated generations (strict invalidation)
- Missing energy values (preservation of null/None, never coerced to 0)
- Zero-token outputs and zero-latency (division-by-zero protection)
- Empty evaluation sets
- Marginal Energy Gain (MEG) calculations
"""

import unittest
from src.evaluation.metrics import (
    compute_experiment_metrics,
    compute_strategy_summary,
    calculate_meg,
    _is_valid_num
)


class TestMetrics(unittest.TestCase):

    def test_empty_records(self):
        """Tests that empty record list returns empty dict and does not error."""
        res = compute_experiment_metrics([])
        self.assertEqual(res, {})

        strat_res = compute_strategy_summary([])
        self.assertEqual(strat_res, {})

    def test_correct_and_incorrect_answers(self):
        """Tests accuracy computation with mixed correct and incorrect answers."""
        records = [
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": False,
                "generation_stop_reason": "stop",
                "answer_parse_success": True,
                "answer_correct": True,
                "exact_match": True,
                "output_tokens": 10,
                "input_tokens": 20,
                "total_tokens": 30,
                "generation_latency_ms": 100.0,
                "total_latency_ms": 120.0,
                "ttft_ms": 20.0,
                "energy_total_j": 10.0
            },
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": False,
                "generation_stop_reason": "stop",
                "answer_parse_success": True,
                "answer_correct": False,
                "exact_match": False,
                "output_tokens": 10,
                "input_tokens": 20,
                "total_tokens": 30,
                "generation_latency_ms": 100.0,
                "total_latency_ms": 120.0,
                "ttft_ms": 20.0,
                "energy_total_j": 10.0
            }
        ]

        summary = compute_experiment_metrics(records)
        self.assertEqual(summary["requested_samples"], 2)
        self.assertEqual(summary["correct_answers"], 1)
        self.assertEqual(summary["total_accuracy"], 0.5)
        self.assertEqual(summary["valid_accuracy"], 0.5)
        self.assertEqual(summary["valid_samples"], 2)
        self.assertEqual(summary["truncated_generations"], 0)
        self.assertEqual(summary["parse_success_count"], 2)
        self.assertEqual(summary["exact_match_answers"], 1)
        self.assertEqual(summary["mean_energy_j"], 10.0)

    def test_strict_truncation_handling(self):
        """Tests that truncated records are excluded from valid-answer counts and accuracy."""
        records = [
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": True,
                "generation_stop_reason": "length",
                "answer_parse_success": False,
                "answer_correct": False,
                "exact_match": False,
                "output_tokens": 512,
                "input_tokens": 50,
                "total_tokens": 562,
                "generation_latency_ms": 2000.0,
                "total_latency_ms": 2100.0,
                "ttft_ms": 100.0,
                "energy_total_j": 50.0
            },
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": False,
                "generation_stop_reason": "stop",
                "answer_parse_success": True,
                "answer_correct": True,
                "exact_match": True,
                "output_tokens": 10,
                "input_tokens": 50,
                "total_tokens": 60,
                "generation_latency_ms": 200.0,
                "total_latency_ms": 300.0,
                "ttft_ms": 100.0,
                "energy_total_j": 10.0
            }
        ]

        summary = compute_experiment_metrics(records)
        self.assertEqual(summary["requested_samples"], 2)
        self.assertEqual(summary["truncated_generations"], 1)
        self.assertEqual(summary["truncation_rate"], 0.5)
        self.assertEqual(summary["valid_samples"], 1)
        self.assertEqual(summary["invalid_samples"], 1)
        # Total accuracy is 1 / 2 = 0.5
        self.assertEqual(summary["total_accuracy"], 0.5)
        # Valid-only accuracy is 1 / 1 = 1.0
        self.assertEqual(summary["valid_accuracy"], 1.0)

    def test_missing_and_nan_energy_values(self):
        """Tests that missing (None) and NaN energy values are preserved as null and not coerced to 0."""
        records = [
            {
                "status": "success",
                "strategy": "few_shot_3",
                "generation_truncated": False,
                "answer_parse_success": True,
                "answer_correct": True,
                "output_tokens": 20,
                "energy_total_j": None,
                "energy_prefill_j": None,
                "energy_decode_j": None,
                "total_latency_ms": 500.0
            },
            {
                "status": "success",
                "strategy": "few_shot_3",
                "generation_truncated": False,
                "answer_parse_success": True,
                "answer_correct": True,
                "output_tokens": 20,
                "energy_total_j": float("nan"),
                "energy_prefill_j": None,
                "energy_decode_j": None,
                "total_latency_ms": 500.0
            }
        ]

        summary = compute_experiment_metrics(records)
        self.assertIsNone(summary["mean_energy_j"])
        self.assertIsNone(summary["total_energy_j"])
        self.assertIsNone(summary["energy_per_correct_answer_j"])
        self.assertIsNone(summary["energy_per_output_token_j"])
        self.assertIsNone(summary["accuracy_per_joule"])

    def test_zero_tokens_and_zero_latency_division_safety(self):
        """Tests that zero tokens and zero latency do not cause DivisionByZero or NaN crashes."""
        records = [
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": False,
                "answer_parse_success": False,
                "answer_correct": False,
                "output_tokens": 0,
                "input_tokens": 0,
                "total_tokens": 0,
                "generation_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "ttft_ms": 0.0,
                "energy_total_j": 0.0
            }
        ]

        summary = compute_experiment_metrics(records)
        self.assertEqual(summary["total_accuracy"], 0.0)
        self.assertIsNone(summary["valid_accuracy"])
        self.assertIsNone(summary["tokens_per_second"])
        self.assertIsNone(summary["energy_per_output_token_j"])
        self.assertIsNone(summary["latency_per_output_token_ms"])

    def test_meg_calculation_and_edge_cases(self):
        """Tests Marginal Energy Gain calculation with normal, zero-delta, and invalid inputs."""
        # Standard positive gain: 0.2 acc gain over 10 J -> 0.02 / J
        meg = calculate_meg(acc_base=0.5, acc_candidate=0.7, energy_base=10.0, energy_candidate=20.0)
        self.assertAlmostEqual(meg, 0.02)

        # Zero energy delta -> returns None
        meg_zero_energy = calculate_meg(acc_base=0.5, acc_candidate=0.7, energy_base=10.0, energy_candidate=10.0)
        self.assertIsNone(meg_zero_energy)

        # Missing energy base -> returns None
        meg_missing = calculate_meg(acc_base=0.5, acc_candidate=0.7, energy_base=None, energy_candidate=20.0)
        self.assertIsNone(meg_missing)

        # NaN energy -> returns None
        meg_nan = calculate_meg(acc_base=0.5, acc_candidate=0.7, energy_base=float("nan"), energy_candidate=20.0)
        self.assertIsNone(meg_nan)


if __name__ == "__main__":
    unittest.main()
