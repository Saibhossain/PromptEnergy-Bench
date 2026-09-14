"""Unit tests for research pipeline extensions, schema reproducibility, and hardware comparison."""

import json
import os
import unittest
import tempfile
import shutil

from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.evaluation.base import EvaluationStatus
from src.monitoring.energy import CodeCarbonMonitor, NullEnergyMonitor, AppleSiliconHardwareMonitor
from src.analysis.tables import generate_research_summary_tables
from scripts.run_all_experiments import resolve_models_list
from scripts.compare_hardware_results import generate_cross_hardware_energy_ratios


class TestResearchPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gsm8k_error_classification(self):
        # 1. Correct
        res_correct = GSM8KEvaluator.evaluate("Answer is #### 42", "42")
        d_corr = res_correct.to_dict()
        self.assertTrue(d_corr["correctness"])
        self.assertEqual(d_corr["error_category"], "none")
        self.assertEqual(d_corr["extracted_prediction"], "42")

        # 2. Arithmetic Error
        res_wrong = GSM8KEvaluator.evaluate("Answer is #### 99", "42")
        d_wrong = res_wrong.to_dict()
        self.assertFalse(d_wrong["correctness"])
        self.assertEqual(d_wrong["error_category"], "arithmetic_error")
        self.assertEqual(d_wrong["extracted_prediction"], "99")

        # 3. Truncated
        res_trunc = GSM8KEvaluator.evaluate("Answer is #### 42", "42", generation_truncated=True)
        d_trunc = res_trunc.to_dict()
        self.assertFalse(d_trunc["correctness"])
        self.assertEqual(d_trunc["error_category"], "generation_truncated")
        self.assertIsNone(d_trunc["extracted_prediction"])

        # 4. Empty
        res_empty = GSM8KEvaluator.evaluate("", "42")
        d_empty = res_empty.to_dict()
        self.assertFalse(d_empty["correctness"])
        self.assertEqual(d_empty["error_category"], "empty_output")

        # 5. Parse Failure
        res_unparseable = GSM8KEvaluator.evaluate("I cannot calculate this right now.", "42")
        d_unparse = res_unparseable.to_dict()
        self.assertFalse(d_unparse["correctness"])
        self.assertEqual(d_unparse["error_category"], "parse_failure")

    def test_model_resolution(self):
        # Specific model
        resolved = resolve_models_list("qwen3.5:0.8b")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["name"], "qwen3.5:0.8b")

        # Multiple comma-separated
        resolved_multi = resolve_models_list("gemma3:4b,mistral:7b")
        self.assertEqual(len(resolved_multi), 2)

        # All models
        resolved_all = resolve_models_list("all")
        self.assertGreaterEqual(len(resolved_all), 5)

    def test_cross_hardware_ratio_calculation(self):
        hw_records = {
            "MacBook Air M1": [
                {"model": "qwen3.5:0.8b", "strategy": "zero_shot_direct", "energy_total_j": 100.0},
                {"model": "qwen3.5:0.8b", "strategy": "zero_shot_direct", "energy_total_j": 120.0}
            ],
            "Windows 10-Core PC": [
                {"model": "qwen3.5:0.8b", "strategy": "zero_shot_direct", "energy_total_j": 200.0},
                {"model": "qwen3.5:0.8b", "strategy": "zero_shot_direct", "energy_total_j": 240.0}
            ]
        }
        out_tables = os.path.join(self.temp_dir, "tables")
        generate_cross_hardware_energy_ratios(hw_records, out_tables)
        ratio_csv = os.path.join(out_tables, "cross_hardware_energy_ratios.csv")
        self.assertTrue(os.path.exists(ratio_csv))

    def test_research_summary_tables_generation(self):
        # Create mock results.jsonl in temp_dir
        run_dir = os.path.join(self.temp_dir, "primary_exp_gsm8k", "windows_pc", "2026-09-14_10-00-00")
        os.makedirs(run_dir, exist_ok=True)
        rec = {
            "experiment_id": "test_run_01",
            "dataset": "gsm8k",
            "model": "qwen3.5:0.8b",
            "strategy": "short_cot",
            "hardware_identifier": "windows_pc",
            "energy_total_j": 500.0,
            "ttft_ms": 250.0,
            "total_latency_ms": 1500.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "answer_correct": True
        }
        with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        analysis_dir = os.path.join(self.temp_dir, "analysis")
        generated = generate_research_summary_tables(self.temp_dir, analysis_dir)
        self.assertEqual(len(generated), 8)
        self.assertTrue(os.path.exists(os.path.join(analysis_dir, "prompt_strategy_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(analysis_dir, "pareto_frontier_summary.csv")))


if __name__ == "__main__":
    unittest.main()
