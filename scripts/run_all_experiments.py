#!/usr/bin/env python3
"""Master Experiment Pipeline Runner for PromptEnergy-Bench.

Sequentially executes benchmark experiments across datasets and models:
1. Experiment 1: Prompting Strategy Comparison (0-shot, Few-shot, CoT, Role-play, System-prompted)
2. Experiment 2: Controlled Context-Length Scaling (0 to 8192 tokens)
3. Experiment 3: Retrieval-Augmented Generation (BM25 RAG) Energy Decomposition

Tailored for MacBook Air M1 8GB RAM with default models:
- qwen3.5:0.8b-mlx
- qwen3.5:2b-mlx

Datasets supported:
- gsm8k (GSM8K Math Reasoning)
- natural_questions (Factoid QA)
- contexteval (Long Context QA)
- cnn_dailymail (Multi-document Summarization)

Supports:
--hardware, --models, --datasets/--dataset, --eval-size, --runs-per-condition,
--output-dir, --seed, --skip-existing, --experiment, --compare
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.device import collect_device_info, normalize_device_name
from src.infrastructure.logging import setup_experiment_logging
from src.experiments.primary import PrimaryExperiment
from src.experiments.context_scaling import ContextScalingExperiment
from src.experiments.rag import RAGExperiment


ALL_DATASETS = ["gsm8k", "natural_questions", "contexteval", "cnn_dailymail"]


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_models_list(
    models_arg: str,
    models_cfg_path: str = "configs/models.yaml"
) -> List[Dict[str, Any]]:
    """Resolves list of model configurations from CLI arg and models.yaml."""
    cfg = load_yaml_config(models_cfg_path)
    registered_models = cfg.get("models", [])
    reg_map = {m["name"]: m for m in registered_models}

    if models_arg.strip().lower() == "all":
        return registered_models

    requested_names = [m.strip() for m in models_arg.split(",") if m.strip()]
    resolved = []
    for name in requested_names:
        if name in reg_map:
            resolved.append(reg_map[name])
        else:
            # Fallback default configuration for specified model
            backend = "openai" if name.startswith(("gpt-", "o1-", "claude-")) else "ollama"
            fmt = "mlx" if "mlx" in name.lower() else ("api" if backend == "openai" else "gguf")
            resolved.append({
                "name": name,
                "backend": backend,
                "format": fmt,
                "context_limit": 4096 if "mlx" in name.lower() else 8192,
                "supports_thinking": "qwen3" in name or "r1" in name
            })
    return resolved


def resolve_datasets_list(dataset_arg: str) -> List[str]:
    """Resolves list of dataset names from CLI arg."""
    raw = dataset_arg.strip().lower()
    if raw in ("all", "full"):
        return list(ALL_DATASETS)
    names = [d.strip() for d in raw.split(",") if d.strip()]
    resolved = []
    for name in names:
        if name in ("nq", "natural_questions", "natural-questions"):
            resolved.append("natural_questions")
        elif name in ("context", "contexteval", "context_eval"):
            resolved.append("contexteval")
        elif name in ("cnn", "cnn_dailymail", "cnn-dailymail"):
            resolved.append("cnn_dailymail")
        elif name in ("gsm", "gsm8k"):
            resolved.append("gsm8k")
        else:
            resolved.append(name)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Master Experiment Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="macbook_air_m1",
        help="Hardware preset name (e.g. macbook_air_m1, windows_10core_pc, generic_cuda_server) or 'auto'"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3.5:0.8b-mlx,qwen3.5:2b-mlx",
        help="Model name, comma-separated model list (e.g. 'qwen3.5:0.8b-mlx,qwen3.5:2b-mlx'), or 'all'"
    )
    parser.add_argument(
        "--dataset", "--datasets",
        dest="dataset",
        type=str,
        default="gsm8k",
        help="Target benchmark dataset name(s), comma-separated list, or 'all' for [gsm8k, natural_questions, contexteval, cnn_dailymail]"
    )
    parser.add_argument(
        "--eval-size",
        type=str,
        default="5",
        help="Evaluation sample size per dataset ('5', '20', '50', '100', 'full')"
    )
    parser.add_argument(
        "--runs-per-condition",
        type=int,
        default=1,
        help="Number of repetitions per condition"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation question sampling"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip conditions that already exist in checkpoints or resume existing runs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume directory path, or 'auto' to automatically find and resume the latest matching incomplete run"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "primary", "context", "rag", "1", "2", "3"],
        help="Which experiment(s) to run: 'all', 'primary' (Exp 1), 'context' (Exp 2), 'rag' (Exp 3)"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[0, 512, 1024, 2048, 4096],
        help="Target context lengths for Experiment 2"
    )
    parser.add_argument(
        "--top-k-list",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k retrieval values for Experiment 3 (RAG)"
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of initial warmup runs before measurement"
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Override inference backend operator (e.g., 'ollama', 'openai', 'transformers')"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Automatically generate comparative tables and figures using scripts/compare_results.py upon completion"
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Disable automatic post-run comparison generation"
    )

    args = parser.parse_args()

    # Parse eval size
    eval_size_val = args.eval_size
    if eval_size_val.isdigit():
        eval_size_val = int(eval_size_val)

    # Normalize hardware preset
    hw_name = args.hardware
    if hw_name.lower() in ("macbook_m1", "macbook", "m1"):
        hw_name = "macbook_air_m1"
    elif hw_name.lower() in ("windows", "windows_x86", "pc"):
        hw_name = "windows_10core_pc"

    models_to_run = resolve_models_list(args.models)
    datasets_to_run = resolve_datasets_list(args.dataset)

    print("=" * 75)
    print("PROMPTENERGY-BENCH: MASTER EXPERIMENT PIPELINE RUNNER")
    print("=" * 75)
    print(f"Hardware Target       : {hw_name}")
    print(f"Models Selected       : {[m['name'] for m in models_to_run]}")
    print(f"Datasets Selected     : {datasets_to_run}")
    print(f"Evaluation Size       : {eval_size_val} samples/dataset")
    print(f"Runs Per Condition    : {args.runs_per_condition}")
    print(f"Warmups               : {args.warmups}")
    print(f"Target Experiment(s)  : {args.experiment}")
    print(f"Output Root Directory : {args.output_dir}")
    print(f"Auto-Compare Report   : {args.compare}")
    print("=" * 75)

    experiments_to_execute = []
    exp_choice = args.experiment.lower()
    if exp_choice in ("all", "1", "primary"):
        experiments_to_execute.append("primary")
    if exp_choice in ("all", "2", "context"):
        experiments_to_execute.append("context")
    if exp_choice in ("all", "3", "rag"):
        experiments_to_execute.append("rag")

    successful_runs = []
    failed_runs = []
    created_run_dirs = []

    total_matrix_steps = len(models_to_run) * len(datasets_to_run) * len(experiments_to_execute)
    step_idx = 0

    for model_info in models_to_run:
        model_name = model_info["name"]
        operator = args.operator or model_info.get("backend", "ollama")
        model_fmt = model_info.get("format", "mlx" if "mlx" in model_name else "gguf")

        for dataset_name in datasets_to_run:
            print(f"\n{'='*75}")
            print(f">>> [Matrix] Model: {model_name} | Dataset: {dataset_name} | Backend: {operator} <<<")
            print(f"{'='*75}")

            base_cli_args = {
                "device_name": hw_name,
                "model": model_name,
                "dataset": dataset_name,
                "operator": operator,
                "model_format": model_fmt,
                "eval_size": eval_size_val,
                "warmups": args.warmups,
                "repetitions": args.runs_per_condition,
                "seed": args.seed,
                "skip_existing": args.skip_existing,
                "resume": args.resume,
                "interactive": False,
                "non_interactive": True
            }

            # 1. Experiment 1: Prompting Strategy Comparison
            if "primary" in experiments_to_execute:
                step_idx += 1
                exp_label = f"primary_exp_{dataset_name}"
                print(f"\n--- [{step_idx}/{total_matrix_steps}] Running Experiment 1: Prompting Strategies ({model_name} on {dataset_name}) ---")
                try:
                    exp1 = PrimaryExperiment(
                        cli_args=base_cli_args,
                        interactive=False
                    )
                    summary1 = exp1.run()
                    run_dir = exp1.paths["run_dir"]
                    successful_runs.append((model_name, dataset_name, exp_label, run_dir))
                    created_run_dirs.append(run_dir)
                    print(f"[SUCCESS] Experiment 1 completed -> {run_dir}")
                except Exception as e:
                    print(f"[ERROR] Experiment 1 failed for {model_name} on {dataset_name}: {e}")
                    failed_runs.append((model_name, dataset_name, exp_label, str(e)))

            # 2. Experiment 2: Context-Length Scaling
            if "context" in experiments_to_execute:
                step_idx += 1
                exp_label = f"context_scaling_{dataset_name}"
                print(f"\n--- [{step_idx}/{total_matrix_steps}] Running Experiment 2: Context-Length Scaling ({model_name} on {dataset_name}) ---")
                ctx_args = dict(base_cli_args)
                ctx_args["context_lengths"] = args.context_lengths
                ctx_args["context_type"] = "relevant"
                try:
                    exp2 = ContextScalingExperiment(
                        cli_args=ctx_args,
                        interactive=False
                    )
                    summary2 = exp2.run()
                    run_dir = exp2.paths["run_dir"]
                    successful_runs.append((model_name, dataset_name, exp_label, run_dir))
                    created_run_dirs.append(run_dir)
                    print(f"[SUCCESS] Experiment 2 completed -> {run_dir}")
                except Exception as e:
                    print(f"[ERROR] Experiment 2 failed for {model_name} on {dataset_name}: {e}")
                    failed_runs.append((model_name, dataset_name, exp_label, str(e)))

            # 3. Experiment 3: BM25 RAG Energy Decomposition
            if "rag" in experiments_to_execute:
                step_idx += 1
                exp_label = f"rag_{dataset_name}"
                print(f"\n--- [{step_idx}/{total_matrix_steps}] Running Experiment 3: BM25 RAG Pipeline ({model_name} on {dataset_name}) ---")
                rag_args = dict(base_cli_args)
                rag_args["top_k_list"] = args.top_k_list
                rag_args["include_baseline"] = True
                try:
                    exp3 = RAGExperiment(
                        cli_args=rag_args,
                        interactive=False
                    )
                    summary3 = exp3.run()
                    run_dir = exp3.paths["run_dir"]
                    successful_runs.append((model_name, dataset_name, exp_label, run_dir))
                    created_run_dirs.append(run_dir)
                    print(f"[SUCCESS] Experiment 3 completed -> {run_dir}")
                except Exception as e:
                    print(f"[ERROR] Experiment 3 failed for {model_name} on {dataset_name}: {e}")
                    failed_runs.append((model_name, dataset_name, exp_label, str(e)))

    print("\n" + "=" * 75)
    print("ALL EXPERIMENT EXECUTIONS FINISHED")
    print("=" * 75)
    print(f"Total Successful Runs: {len(successful_runs)}")
    for model, ds, exp_type, run_path in successful_runs:
        print(f"  [OK] {model:20s} | {ds:18s} | {exp_type:24s} | {run_path}")

    if failed_runs:
        print(f"\nTotal Failed Runs: {len(failed_runs)}")
        for model, ds, exp_type, err in failed_runs:
            print(f"  [FAILED] {model:20s} | {ds:18s} | {exp_type:24s} | Error: {err}")

    # Post-run automatic comparison report generation
    if args.compare and created_run_dirs:
        print("\n" + "=" * 75)
        print("GENERATING AUTOMATIC COMPARISON REPORT & PUBLICATION FIGURES")
        print("=" * 75)
        report_output_dir = os.path.join(args.output_dir, "master_comparison")
        try:
            compare_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "compare_results.py"),
                "--inputs"
            ] + created_run_dirs + [
                "--output-dir", report_output_dir
            ]
            print(f"Executing: {' '.join(compare_cmd[:4])} ... --output-dir {report_output_dir}")
            subprocess.run(compare_cmd, check=True)
            print(f"\n[SUCCESS] Master Comparison Report generated in: {report_output_dir}")
        except Exception as e:
            print(f"[WARNING] Automatic comparison report generation encountered an error: {e}")

    print("=" * 75)
    print("Run interactive comparison anytime via:")
    print("  python scripts/compare_results.py")
    print("=" * 75)


if __name__ == "__main__":
    main()

