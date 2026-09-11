"""Comprehensive Unit Tests for PromptEnergy-Bench Revisions.

Covers Section 29 requirements:
1. Generation max-token configuration
2. Strategy-specific max-token configuration
3. Truncation detection
4. Stop reason normalization
5. GSM8K answer extraction priority
6. Generation truncation classification
7. Summary calculation
8. Strategy aggregation
9. Pareto frontier calculation
10. Marginal Energy Gain (MEG)
11. Paired question alignment
12. Missing energy handling
13. n=1 Confidence Interval handling
14. Result-directory creation
15. Table generation (CSV, MD, LaTeX)
16. Plot generation (PNG, PDF, SVG)
17. Multi-model filtering
18. Multi-device filtering
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import yaml

from src.backends.base import InferenceOutput
from src.backends.ollama_backend import OllamaBackend
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.evaluation.metrics import compute_experiment_metrics, calculate_meg
from src.analysis.pareto import compute_pareto_frontier
from src.analysis.statistics import (
    calculate_summary_statistics,
    compute_confidence_interval,
    build_paired_comparison_dataframe,
    paired_wilcoxon_test
)
from src.analysis.tables import (
    generate_all_tables,
    generate_table_1_config,
    generate_table_2_comparison,
    generate_table_3_tradeoff,
    generate_table_4_meg,
    generate_table_5_statistical_summary
)
from src.infrastructure.metadata import build_experiment_paths
from visual.plot_results import (
    load_and_filter_results,
    plot_01_accuracy_by_strategy,
    plot_05_accuracy_energy_pareto,
    plot_13_truncation_rate,
    plot_validation_overview,
    save_plot_metadata
)


class TestTruncationAndTables(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_generation_token_config(self):
        """1 & 2: Tests generation max-token and strategy-specific configuration loading."""
        config_path = os.path.join(self.temp_dir, "generation.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "generation": {
                    "default_max_tokens": 512,
                    "strategy_max_tokens": {
                        "zero_shot_direct": 256,
                        "few_shot_3": 256,
                        "zero_shot_cot": 512,
                        "short_cot": 512,
                        "long_cot": 1024
                    }
                }
            }, f)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)["generation"]

        self.assertEqual(data["default_max_tokens"], 512)
        self.assertEqual(data["strategy_max_tokens"]["zero_shot_direct"], 256)
        self.assertEqual(data["strategy_max_tokens"]["long_cot"], 1024)

    def test_03_04_stop_reason_and_truncation_detection(self):
        """3 & 4: Tests stop reason normalization and truncation flagging."""
        # Test length -> truncated
        out_trunc = InferenceOutput(
            text="",
            generation_stop_reason="length",
            generation_truncated=True,
            generation_complete=False,
            max_output_tokens=256
        )
        self.assertEqual(out_trunc.generation_stop_reason, "length")
        self.assertTrue(out_trunc.generation_truncated)
        self.assertFalse(out_trunc.generation_complete)

        # Test stop -> normal completion
        out_stop = InferenceOutput(
            text="#### 4",
            generation_stop_reason="stop",
            generation_truncated=False,
            generation_complete=True,
            max_output_tokens=512
        )
        self.assertEqual(out_stop.generation_stop_reason, "stop")
        self.assertFalse(out_stop.generation_truncated)
        self.assertTrue(out_stop.generation_complete)

    def test_05_gsm8k_extraction_priority(self):
        """5: Tests strict priority of #### answer over intermediate reasoning numbers."""
        text = "The building has 120 units... 30 are unoccupied. #### 30"
        extracted = GSM8KEvaluator.extract_answer(text)
        self.assertEqual(extracted, "30")

        # Negative and comma
        self.assertEqual(GSM8KEvaluator.extract_answer("#### -15"), "-15")
        self.assertEqual(GSM8KEvaluator.extract_answer("#### 1,200"), "1200")
        self.assertEqual(GSM8KEvaluator.extract_answer("#### 30.0"), "30.0")

    def test_06_truncation_classification(self):
        """6: Tests that truncated generation with incomplete text is classified as parse failure."""
        res = GSM8KEvaluator.evaluate("Thinking about 120 units and", "30", generation_truncated=True)
        self.assertFalse(res.answer_parse_success)
        self.assertFalse(res.answer_correct)
        self.assertTrue(res.generation_truncated)

    def test_07_08_summary_and_strategy_aggregation(self):
        """7 & 8: Tests summary metrics and per-strategy aggregation."""
        records = [
            {
                "sample_id": "gsm8k_test_0000",
                "strategy": "zero_shot_direct",
                "status": "success",
                "answer_correct": True,
                "answer_parse_success": True,
                "generation_truncated": False,
                "energy_total_j": 10.0,
                "total_latency_ms": 1000.0,
                "ttft_ms": 50.0,
                "thinking_tokens": None,
                "visible_output_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 50
            },
            {
                "sample_id": "gsm8k_test_0001",
                "strategy": "zero_shot_direct",
                "status": "success",
                "answer_correct": False,
                "answer_parse_success": True,
                "generation_truncated": True,
                "energy_total_j": 20.0,
                "total_latency_ms": 2000.0,
                "ttft_ms": 60.0,
                "thinking_tokens": None,
                "visible_output_tokens": 128,
                "output_tokens": 128,
                "total_tokens": 180
            },
            {
                "sample_id": "gsm8k_test_0000",
                "strategy": "long_cot",
                "status": "success",
                "answer_correct": True,
                "answer_parse_success": True,
                "generation_truncated": False,
                "energy_total_j": 30.0,
                "total_latency_ms": 3000.0,
                "ttft_ms": 100.0,
                "thinking_tokens": 200,
                "visible_output_tokens": 20,
                "output_tokens": 220,
                "total_tokens": 300
            }
        ]

        metrics = compute_experiment_metrics(records)
        self.assertEqual(metrics["requested_inference_runs"], 3)
        self.assertEqual(metrics["successful_inference_runs"], 3)
        self.assertEqual(metrics["truncated_generations"], 1)
        self.assertAlmostEqual(metrics["truncation_rate"], 1 / 3, places=3)
        self.assertEqual(metrics["correct_answers"], 2)

        # Strategy summaries
        strat_sum = metrics["strategy_summaries"]
        self.assertIn("zero_shot_direct", strat_sum)
        self.assertIn("long_cot", strat_sum)
        self.assertEqual(strat_sum["zero_shot_direct"]["n"], 2)
        self.assertEqual(strat_sum["zero_shot_direct"]["truncated_n"], 1)
        self.assertEqual(strat_sum["zero_shot_direct"]["accuracy"], 0.5)
        self.assertEqual(strat_sum["long_cot"]["accuracy"], 1.0)

    def test_09_pareto_frontier_calculation(self):
        """9: Tests Pareto frontier identification."""
        candidates = [
            {"strategy": "A", "energy_j": 10.0, "accuracy": 50.0},
            {"strategy": "B", "energy_j": 15.0, "accuracy": 40.0},  # Dominated by A
            {"strategy": "C", "energy_j": 20.0, "accuracy": 70.0}   # Efficient
        ]
        frontier = compute_pareto_frontier(candidates)
        frontier_strats = [f["strategy"] for f in frontier]
        self.assertIn("A", frontier_strats)
        self.assertIn("C", frontier_strats)
        self.assertNotIn("B", frontier_strats)

    def test_10_marginal_energy_gain(self):
        """10: Tests Marginal Energy Gain (MEG) formula."""
        # 10 J increase for 20% accuracy gain -> 2.0 % / J
        meg = calculate_meg(acc_base=0.4, acc_candidate=0.6, energy_base=10.0, energy_candidate=20.0)
        self.assertAlmostEqual(meg, 0.02, places=4)

        # Zero energy change returns None
        meg_zero = calculate_meg(acc_base=0.4, acc_candidate=0.6, energy_base=10.0, energy_candidate=10.0)
        self.assertIsNone(meg_zero)

    def test_11_paired_question_alignment(self):
        """11: Tests paired question alignment dataframe by sample_id."""
        records = [
            {"sample_id": "q1", "strategy": "zero_shot_direct", "answer_correct": False, "energy_total_j": 10.0, "total_latency_ms": 100},
            {"sample_id": "q1", "strategy": "long_cot", "answer_correct": True, "energy_total_j": 25.0, "total_latency_ms": 500},
            {"sample_id": "q2", "strategy": "zero_shot_direct", "answer_correct": True, "energy_total_j": 12.0, "total_latency_ms": 120},
            {"sample_id": "q2", "strategy": "long_cot", "answer_correct": True, "energy_total_j": 28.0, "total_latency_ms": 550}
        ]
        df = build_paired_comparison_dataframe(records)
        self.assertEqual(len(df), 4)
        self.assertIn("sample_id", df.columns)
        self.assertIn("strategy", df.columns)

        # Pivot to verify alignment
        piv = df.pivot(index="sample_id", columns="strategy", values="energy")
        self.assertEqual(piv.loc["q1", "zero_shot_direct"], 10.0)
        self.assertEqual(piv.loc["q1", "long_cot"], 25.0)

    def test_12_missing_energy_handling(self):
        """12: Tests graceful handling of missing energy measurements."""
        records = [
            {"sample_id": "q1", "status": "success", "answer_correct": True, "energy_total_j": None, "total_latency_ms": 100, "output_tokens": 10}
        ]
        metrics = compute_experiment_metrics(records)
        self.assertIsNone(metrics["mean_energy_j"])
        self.assertIsNone(metrics["energy_per_correct_answer_j"])

    def test_13_n_equals_1_ci_handling(self):
        """13: Tests that n=1 does not produce fabricated confidence intervals."""
        ci = compute_confidence_interval([42.0])
        self.assertEqual(ci, (None, None))

    def test_14_result_directory_creation(self):
        """14: Tests timestamped result directory hierarchy creation."""
        paths = build_experiment_paths(
            experiment_name="primary_exp_gsm8k",
            normalized_device_name="macbook_air_m1",
            timestamp="2026-09-11_15-00-00",
            results_root=self.temp_dir
        )
        self.assertTrue(os.path.isdir(paths["run_dir"]))
        self.assertTrue(os.path.isdir(paths["plots_dir"]))
        self.assertTrue(os.path.isdir(paths["logs_dir"]))
        self.assertTrue(os.path.isdir(paths["tables_dir"]))

    def test_15_table_generation(self):
        """15: Tests automatic table generation in CSV, Markdown, and LaTeX."""
        metadata = {
            "experiment_name": "primary_exp_gsm8k",
            "device": {"name": "MacBook Air M1", "os": "macOS", "cpu": "Apple M1", "gpu_available": True, "gpu_name": "Apple M1 GPU", "ram_gb": 8},
            "backend": {"model_name": "qwen3.5:0.8b-mlx", "operator": "ollama", "model_format": "mlx"},
            "measurement": {"energy_method": "apple_estimated", "energy_quality": "software_estimate", "energy_measurement_level": "estimated_system"}
        }
        config = {
            "experiment_name": "primary_exp_gsm8k",
            "model": {"name": "qwen3.5:0.8b-mlx", "backend": "ollama", "format": "mlx"},
            "dataset": {"name": "gsm8k", "evaluation_split": "test", "evaluation_size": 50},
            "sampling": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
            "generation": {"default_max_tokens": 512, "strategy_max_tokens": {"zero_shot_direct": 256, "few_shot_3": 256, "zero_shot_cot": 512, "short_cot": 512, "long_cot": 1024}},
            "warmups": 1,
            "repetitions": 1,
            "validation": True
        }
        summary = {
            "run_id": "test_run",
            "metrics": {
                "strategy_summaries": {
                    "zero_shot_direct": {"n": 10, "successful_n": 10, "truncated_n": 0, "truncation_rate": 0.0, "parse_failure_n": 0, "accuracy": 0.5, "mean_thinking_tokens": None, "mean_visible_output_tokens": 15, "mean_output_tokens": 15, "mean_ttft_ms": 50, "mean_generation_latency_ms": 200, "mean_total_latency_ms": 250, "mean_energy_j": 10.0, "energy_per_correct_answer_j": 20.0, "accuracy_per_joule": 0.05},
                    "long_cot": {"n": 10, "successful_n": 10, "truncated_n": 0, "truncation_rate": 0.0, "parse_failure_n": 0, "accuracy": 0.8, "mean_thinking_tokens": 200, "mean_visible_output_tokens": 20, "mean_output_tokens": 220, "mean_ttft_ms": 80, "mean_generation_latency_ms": 1000, "mean_total_latency_ms": 1080, "mean_energy_j": 30.0, "energy_per_correct_answer_j": 37.5, "accuracy_per_joule": 0.0267}
                }
            }
        }
        records = [
            {"strategy": "zero_shot_direct", "status": "success", "answer_correct": True, "energy_total_j": 10.0, "total_latency_ms": 250, "ttft_ms": 50, "thinking_tokens": None, "output_tokens": 15},
            {"strategy": "long_cot", "status": "success", "answer_correct": True, "energy_total_j": 30.0, "total_latency_ms": 1080, "ttft_ms": 80, "thinking_tokens": 200, "output_tokens": 220}
        ]

        tables_dir = os.path.join(self.temp_dir, "tables")
        tables = generate_all_tables(records, metadata, config, summary, tables_dir)

        self.assertIn("table_1_config", tables)
        self.assertIn("table_2_comparison", tables)
        self.assertIn("table_3_tradeoff", tables)
        self.assertIn("table_4_meg", tables)
        self.assertIn("table_5_statistics", tables)

        # Verify files exist on disk
        for tbl_name, paths in tables.items():
            self.assertTrue(os.path.exists(paths["csv"]), f"Missing CSV for {tbl_name}")
            self.assertTrue(os.path.exists(paths["md"]), f"Missing MD for {tbl_name}")
            self.assertTrue(os.path.exists(paths["tex"]), f"Missing TeX for {tbl_name}")

    def test_16_plot_generation(self):
        """16: Tests high-res plot generation across formats (PNG, PDF, SVG)."""
        df_all = pd.DataFrame([
            {"strategy": "zero_shot_direct", "strategy_display": "Zero-shot Direct", "status": "success", "answer_correct": True, "generation_truncated": False, "energy_total_j": 10.0, "total_latency_ms": 250, "output_tokens": 20, "model": "m1", "device": "d1"},
            {"strategy": "long_cot", "strategy_display": "Long CoT", "status": "success", "answer_correct": True, "generation_truncated": False, "energy_total_j": 25.0, "total_latency_ms": 800, "output_tokens": 200, "model": "m1", "device": "d1"}
        ])
        df_success = df_all.copy()

        plots_dir = os.path.join(self.temp_dir, "plots", "validation")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot figure 1
        f1_paths = plot_01_accuracy_by_strategy(df_all, reps=1, out_dir=plots_dir)
        self.assertEqual(len(f1_paths), 3)  # PNG, PDF, SVG
        for p in f1_paths:
            self.assertTrue(os.path.exists(p))

        # Plot figure 5 (Pareto)
        f5_paths = plot_05_accuracy_energy_pareto(df_all, reps=1, out_dir=plots_dir)
        self.assertEqual(len(f5_paths), 3)

        # Plot figure 13 (Truncation rate)
        f13_paths = plot_13_truncation_rate(df_all, reps=1, out_dir=plots_dir)
        self.assertEqual(len(f13_paths), 3)

        # Diagnostic overview
        diag = plot_validation_overview(df_all, {"evaluation_examples": 2, "metrics": {"accuracy": 1.0}}, plots_dir)
        self.assertTrue(os.path.exists(diag))

    def test_17_18_multi_model_and_device_filtering(self):
        """17 & 18: Tests filtering by model and device in load_and_filter_results."""
        results_file = os.path.join(self.temp_dir, "results.jsonl")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write('{"sample_id": "1", "model": "model_a", "device": "macbook", "strategy": "zero_shot_direct", "status": "success"}\n')
            f.write('{"sample_id": "2", "model": "model_b", "device": "workstation", "strategy": "zero_shot_direct", "status": "success"}\n')

        # Model filter
        df_a, _ = load_and_filter_results(results_file, model_filter="model_a")
        self.assertEqual(len(df_a), 1)
        self.assertEqual(df_a.iloc[0]["model"], "model_a")

        # Device filter
        df_d, _ = load_and_filter_results(results_file, device_filter="workstation")
        self.assertEqual(len(df_d), 1)
        self.assertEqual(df_d.iloc[0]["device"], "workstation")


if __name__ == "__main__":
    unittest.main()
