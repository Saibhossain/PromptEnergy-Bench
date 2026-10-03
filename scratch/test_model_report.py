import os
import sys
import shutil
import json
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.analysis.model_report import generate_single_model_report, get_clean_model_tag

def test_model_report_generation():
    print("Testing clean model tags...")
    assert get_clean_model_tag("qwen3.5:0.8b") == "qwen08B"
    assert get_clean_model_tag("qwen3.5:2b") == "qwen2B"
    assert get_clean_model_tag("gemma3:4b") == "gemma4B"
    assert get_clean_model_tag("qwen3.5:2b-mlx") == "qwen2B"
    print("Model tags test passed!")

    # Create dummy model experiment folder with realistic sub-experiment data
    test_dir = "results/test_hardware/exp_1_qwen2B_10"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Sub-experiment 1: GSM8K Primary
    exp1_dir = os.path.join(test_dir, "primary_exp_gsm8k")
    os.makedirs(exp1_dir, exist_ok=True)
    records_1 = [
        {
            "dataset": "gsm8k",
            "strategy": "zero_shot_direct",
            "sample_id": f"gsm_{i}",
            "answer_correct": i % 2 == 0,
            "energy_total_j": 12.5 + i * 0.1,
            "energy_net_j": 10.0 + i * 0.1,
            "energy_prefill_j": 2.0,
            "energy_decode_j": 8.0,
            "total_latency_ms": 450.0 + i * 5,
            "generation_latency_ms": 400.0,
            "ttft_ms": 50.0,
            "input_tokens": 85,
            "output_tokens": 60,
            "thinking_tokens": None,
            "active_power_w": 25.0,
            "idle_power_w": 5.0,
            "metric_values": {"exact_match": i % 2 == 0}
        }
        for i in range(10)
    ]
    with open(os.path.join(exp1_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in records_1:
            f.write(json.dumps(r) + "\n")

    # Sub-experiment 2: Context Scaling Natural Questions
    exp2_dir = os.path.join(test_dir, "context_scaling_natural_questions")
    os.makedirs(exp2_dir, exist_ok=True)
    records_2 = [
        {
            "dataset": "natural_questions",
            "strategy": f"ctx_{ctx_len}",
            "sample_id": f"nq_{i}_{ctx_len}",
            "answer_correct": True,
            "energy_total_j": 15.0 + ctx_len * 0.01,
            "energy_net_j": 12.0 + ctx_len * 0.01,
            "energy_prefill_j": 3.0 + ctx_len * 0.005,
            "energy_decode_j": 9.0,
            "total_latency_ms": 500.0 + ctx_len * 0.2,
            "generation_latency_ms": 400.0,
            "ttft_ms": 100.0 + ctx_len * 0.2,
            "input_tokens": ctx_len + 30,
            "output_tokens": 45,
            "thinking_tokens": None,
            "active_power_w": 28.0,
            "idle_power_w": 5.0,
            "metric_values": {"token_f1": 0.85, "span_match": True}
        }
        for ctx_len in [0, 512, 1024, 2048, 4096]
        for i in range(5)
    ]
    with open(os.path.join(exp2_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in records_2:
            f.write(json.dumps(r) + "\n")

    # Sub-experiment 3: RAG CNN/DailyMail
    exp3_dir = os.path.join(test_dir, "rag_cnn_dailymail")
    os.makedirs(exp3_dir, exist_ok=True)
    records_3 = [
        {
            "dataset": "cnn_dailymail",
            "strategy": f"rag_top_{k}",
            "sample_id": f"cnn_{i}_{k}",
            "answer_correct": True,
            "energy_total_j": 20.0 + k * 2.0,
            "energy_net_j": 17.0 + k * 2.0,
            "energy_prefill_j": 5.0 + k * 1.0,
            "energy_decode_j": 12.0,
            "total_latency_ms": 800.0 + k * 50,
            "generation_latency_ms": 650.0,
            "ttft_ms": 150.0 + k * 50,
            "input_tokens": 500 + k * 150,
            "output_tokens": 120,
            "thinking_tokens": None,
            "active_power_w": 30.0,
            "idle_power_w": 5.0,
            "metric_values": {"rougeL_f1": 0.42 + k * 0.01}
        }
        for k in [1, 3, 5]
        for i in range(5)
    ]
    with open(os.path.join(exp3_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in records_3:
            f.write(json.dumps(r) + "\n")

    # Metadata
    with open(os.path.join(test_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"measurement": {"energy_method": "nvidia_nvml"}}, f)

    print("\nRunning generate_single_model_report...")
    report = generate_single_model_report(
        model_run_dir=test_dir,
        model_name="qwen3.5:2b",
        hw_name="test_hardware"
    )

    # Validate output files exist
    assert os.path.exists(os.path.join(test_dir, "tables", "table1_full_merged_results.csv")), "Table 1 CSV missing"
    assert os.path.exists(os.path.join(test_dir, "tables", "table1_full_merged_results.md")), "Table 1 MD missing"
    assert os.path.exists(os.path.join(test_dir, "tables", "table1_full_merged_results.tex")), "Table 1 TeX missing"
    assert os.path.exists(os.path.join(test_dir, "tables", "table2_model_execution_energy.csv")), "Table 2 CSV missing"
    assert os.path.exists(os.path.join(test_dir, "tables", "table2_model_execution_energy.md")), "Table 2 MD missing"
    assert os.path.exists(os.path.join(test_dir, "summary_all_experiments.json")), "summary_all_experiments.json missing"
    
    plots = os.listdir(os.path.join(test_dir, "plots"))
    print("Generated Plots:", plots)
    assert len(plots) >= 7, f"Expected at least 7 plots, found {len(plots)}"

    # Clean up test dir
    shutil.rmtree(test_dir)
    print("\nALL SINGLE MODEL REPORT & SEABORN TESTS PASSED WITH 100% SUCCESS!")

if __name__ == "__main__":
    test_model_report_generation()
