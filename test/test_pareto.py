"""Unit tests for Pareto frontier analysis and budget-constrained strategy selection."""

import unittest
from src.analysis.pareto import is_pareto_efficient, find_pareto_frontier
from src.analysis.selection import select_prompt_strategy


class TestParetoAndSelection(unittest.TestCase):

    def setUp(self):
        # Sample strategies:
        # A: Acc 0.50, Energy 2.0 J, Latency 200 ms (Fast & Low Energy, Low Acc)
        # B: Acc 0.75, Energy 5.0 J, Latency 500 ms (Balanced)
        # C: Acc 0.85, Energy 12.0 J, Latency 1200 ms (High Acc, High Energy)
        # D: Acc 0.70, Energy 6.0 J, Latency 600 ms (Dominated by B!)
        self.configs = [
            {"strategy": "strat_A", "accuracy": 0.50, "mean_energy_j": 2.0, "mean_latency_ms": 200.0},
            {"strategy": "strat_B", "accuracy": 0.75, "mean_energy_j": 5.0, "mean_latency_ms": 500.0},
            {"strategy": "strat_C", "accuracy": 0.85, "mean_energy_j": 12.0, "mean_latency_ms": 1200.0},
            {"strategy": "strat_D", "accuracy": 0.70, "mean_energy_j": 6.0, "mean_latency_ms": 600.0}
        ]

    def test_pareto_dominance(self):
        # D is strictly worse than B in accuracy (0.70 < 0.75), energy (6.0 > 5.0), latency (600 > 500)
        self.assertFalse(is_pareto_efficient(self.configs[3], self.configs))
        # A, B, C should be Pareto efficient
        self.assertTrue(is_pareto_efficient(self.configs[0], self.configs))
        self.assertTrue(is_pareto_efficient(self.configs[1], self.configs))
        self.assertTrue(is_pareto_efficient(self.configs[2], self.configs))

        frontier = find_pareto_frontier(self.configs)
        frontier_names = [f["strategy"] for f in frontier]
        self.assertIn("strat_A", frontier_names)
        self.assertIn("strat_B", frontier_names)
        self.assertIn("strat_C", frontier_names)
        self.assertNotIn("strat_D", frontier_names)

    def test_budget_constrained_selection(self):
        # Target: accuracy >= 0.70, energy <= 6.0 J, latency <= 1000 ms
        best_energy = select_prompt_strategy(
            self.configs,
            accuracy_target=0.70,
            energy_budget=6.0,
            latency_budget=1000.0,
            optimization_priority="energy"
        )
        # Between B (0.75 acc, 5J) and D (0.70 acc, 6J), B is chosen because 5J < 6J
        self.assertIsNotNone(best_energy)
        self.assertEqual(best_energy["strategy"], "strat_B")

        # Infeasible constraint: accuracy >= 0.90
        infeasible = select_prompt_strategy(self.configs, accuracy_target=0.90)
        self.assertIsNone(infeasible)


if __name__ == "__main__":
    unittest.main()
