"""Unit Tests for Universal Green AI and Quality Metric Computation.

Tests all mathematical formulations from research_paper.md:
- MEG, MAG_token, MAG_latency, Pareto & Budget-constrained selection.
- Denominator safety, zero-sample edge cases, and validity statuses.
- Strict null preservation for unmeasured energy and token values.
"""

import unittest
from src.evaluation.metrics import (
    calculate_meg,
    calculate_mag_token,
    calculate_mag_latency,
    solve_budget_constrained_prompting,
    compute_strategy_summary,
    compute_experiment_metrics
)
from src.evaluation.base import MetricValidityStatus, EvaluationStatus


class TestUniversalMetrics(unittest.TestCase):
    """Tests Green AI, quality, and marginal efficiency metrics."""

    def test_meg_calculation(self):
        """Tests Marginal Energy Gain calculation: MEG = [A(B2) - A(B1)] / [E(B2) - E(B1)]."""
        # Baseline: 50% accuracy (0.50), 2.0 Joules
        # Candidate: 70% accuracy (0.70), 3.0 Joules
        # Delta Acc: +0.20, Delta Energy: +1.0 Joule
        # MEG = 0.20 / 1.0 = 0.20 / J
        meg = calculate_meg(0.50, 0.70, 2.0, 3.0)
        self.assertAlmostEqual(meg, 0.20, places=4)

        # None inputs or zero energy difference
        self.assertIsNone(calculate_meg(None, 0.70, 2.0, 3.0))
        self.assertIsNone(calculate_meg(0.50, 0.70, 2.0, 2.0))

    def test_mag_token_and_latency(self):
        """Tests MAG per token and per second."""
        # 0.10 gain for 100 additional tokens -> 0.001 / tok
        mag_tok = calculate_mag_token(0.50, 0.60, 50.0, 150.0)
        self.assertAlmostEqual(mag_tok, 0.001, places=4)

        # 0.10 gain for 1000 ms (1.0 s) -> 0.10 / s
        mag_lat = calculate_mag_latency(0.50, 0.60, 1000.0, 2000.0)
        self.assertAlmostEqual(mag_lat, 0.10, places=2)

    def test_budget_constrained_selection(self):
        """Tests solving: max A(pi) s.t. E(pi) <= B_E, T(pi) <= B_T."""
        candidates = [
            {"strategy": "zero_shot", "accuracy": 0.50, "energy_j": 1.0, "latency_ms": 200.0},
            {"strategy": "few_shot", "accuracy": 0.65, "energy_j": 2.5, "latency_ms": 500.0},
            {"strategy": "cot", "accuracy": 0.85, "energy_j": 6.0, "latency_ms": 1500.0}
        ]

        # Budget: 3.0 Joules, 1000 ms -> Should select few_shot (0.65 acc)
        best = solve_budget_constrained_prompting(candidates, max_energy_j=3.0, max_latency_ms=1000.0)
        self.assertIsNotNone(best)
        self.assertEqual(best["strategy"], "few_shot")

        # Strict budget: 1.5 Joules -> Should select zero_shot (0.50 acc)
        best_strict = solve_budget_constrained_prompting(candidates, max_energy_j=1.5)
        self.assertEqual(best_strict["strategy"], "zero_shot")

        # Impossible budget: 0.5 Joules -> Returns None
        best_none = solve_budget_constrained_prompting(candidates, max_energy_j=0.5)
        self.assertIsNone(best_none)

    def test_all_truncated_records(self):
        """Tests metrics behavior when 100% of samples are truncated."""
        records = [
            {
                "status": "success",
                "strategy": "long_cot",
                "generation_truncated": True,
                "generation_stop_reason": "length",
                "evaluation_status": EvaluationStatus.GENERATION_TRUNCATED.value,
                "answer_correct": False,
                "parse_success": False,
                "output_tokens": 1024,
                "thinking_tokens": 1024,
                "visible_output_tokens": 0,
                "total_tokens": 1024,
                "energy_total_j": 3.5,
                "generation_latency_ms": 1200.0,
                "total_latency_ms": 1300.0
            }
            for _ in range(10)
        ]

        summary = compute_strategy_summary(records)
        self.assertEqual(summary["n"], 10)
        self.assertEqual(summary["truncated_n"], 10)
        self.assertEqual(summary["truncation_rate"], 1.0)
        self.assertEqual(summary["valid_n"], 0)
        self.assertEqual(summary["total_accuracy"], 0.0)
        self.assertIsNone(summary["valid_accuracy"])
        self.assertEqual(summary["metric_validity_status"], MetricValidityStatus.INSUFFICIENT_VALID_SAMPLES.value)

    def test_missing_energy_and_tokens_preservation(self):
        """Ensures missing energy and phase metrics remain None (null) and not 0.0."""
        records = [
            {
                "status": "success",
                "strategy": "zero_shot_direct",
                "generation_truncated": False,
                "generation_stop_reason": "stop",
                "evaluation_status": EvaluationStatus.COMPLETED_CORRECT.value,
                "answer_correct": True,
                "answer_parse_success": True,
                "parse_success": True,
                "output_tokens": 30,
                "thinking_tokens": None,
                "visible_output_tokens": 30,
                "total_tokens": 30,
                "energy_total_j": None,
                "energy_prefill_j": None,
                "energy_decode_j": None,
                "energy_net_j": None,
                "idle_power_w": None,
                "generation_latency_ms": 150.0,
                "total_latency_ms": 160.0
            }
        ]

        summary = compute_strategy_summary(records)
        self.assertEqual(summary["total_accuracy"], 1.0)
        self.assertEqual(summary["valid_accuracy"], 1.0)
        self.assertIsNone(summary["mean_energy_j"])
        self.assertIsNone(summary["total_energy_j"])
        self.assertIsNone(summary["mean_prefill_energy_j"])
        self.assertIsNone(summary["mean_decode_energy_j"])
        self.assertIsNone(summary["mean_net_energy_j"])
        self.assertIsNone(summary["mean_idle_power_w"])
        self.assertIsNone(summary["mean_thinking_tokens"])


if __name__ == "__main__":
    unittest.main()
