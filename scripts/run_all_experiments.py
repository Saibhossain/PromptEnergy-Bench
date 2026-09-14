#!/usr/bin/env python3
"""Master Experiment Runner for PromptEnergy-Bench.

Sequentially executes:
1. Experiment 1: Prompting Strategy Comparison
2. Experiment 2: Controlled Context-Length Scaling (0 to 8192 tokens)
3. Experiment 3: Retrieval-Augmented Generation (BM25 RAG) Energy Decomposition

Supports:
--hardware, --models, --dataset, --eval-size, --runs-per-condition, --output-dir,
--seed, --skip-existing, --experiment
"""

import argparse
import os
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
            fmt = "api" if backend == "openai" else "gguf"
            resolved.append({
                "name": name,
                "backend": backend,
                "format": fmt,
                "context_limit": 8192,
                "supports_thinking": "qwen3" in name or "r1" in name
            })
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Master Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="auto",
        help="Hardware preset name (e.g. windows_10core_pc, macbook_air_m1, macbook_m1, generic_cuda_server) or 'auto'"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3.5:0.8b",
        help="Model name, comma-separated model list, or 'all' to run all registered models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Target benchmark dataset name"
    )
    parser.add_argument(
        "--eval-size",
        type=str,
        default="5",
        help="Evaluation sample size ('5', '50', '100', 'full')"
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
        help="Skip conditions that already exist in checkpoints"
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
        default=[0, 512, 1024, 2048, 4096, 8192],
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

    print("=" * 70)
    print("PROMPTENERGY-BENCH: MASTER EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Hardware Target       : {hw_name}")
    print(f"Models Selected       : {[m['name'] for m in models_to_run]}")
    print(f"Dataset               : {args.dataset}")
    print(f"Evaluation Size       : {eval_size_val}")
    print(f"Runs Per Condition    : {args.runs_per_condition}")
    print(f"Warmups               : {args.warmups}")
    print(f"Target Experiment(s)  : {args.experiment}")
    print(f"Output Root Directory : {args.output_dir}")
    print("=" * 70)

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

    for model_info in models_to_run:
        model_name = model_info["name"]
        operator = args.operator or model_info.get("backend", "ollama")
        model_fmt = model_info.get("format", "gguf")

        print(f"\n>>> Starting Benchmark Suite for Model: {model_name} (Backend: {operator}, Format: {model_fmt}) <<<")

        base_cli_args = {
            "device_name": hw_name,
            "model": model_name,
            "operator": operator,
            "model_format": model_fmt,
            "eval_size": eval_size_val,
            "warmups": args.warmups,
            "repetitions": args.runs_per_condition,
            "seed": args.seed,
            "skip_existing": args.skip_existing,
            "interactive": False,
            "non_interactive": True
        }

        # 1. Experiment 1: Prompting Strategy Comparison
        if "primary" in experiments_to_execute:
            print(f"\n--- [1/3] Running Experiment 1: Prompting Strategies ({model_name}) ---")
            try:
                exp1 = PrimaryExperiment(
                    cli_args=base_cli_args,
                    interactive=False
                )
                summary1 = exp1.run()
                successful_runs.append((model_name, "primary_exp_gsm8k", exp1.paths["run_dir"]))
                print(f"[SUCCESS] Experiment 1 completed -> {exp1.paths['run_dir']}")
            except Exception as e:
                print(f"[ERROR] Experiment 1 failed for {model_name}: {e}")
                failed_runs.append((model_name, "primary_exp_gsm8k", str(e)))

        # 2. Experiment 2: Context-Length Scaling
        if "context" in experiments_to_execute:
            print(f"\n--- [2/3] Running Experiment 2: Context-Length Scaling ({model_name}) ---")
            ctx_args = dict(base_cli_args)
            ctx_args["context_lengths"] = args.context_lengths
            ctx_args["context_type"] = "relevant"
            try:
                exp2 = ContextScalingExperiment(
                    cli_args=ctx_args,
                    interactive=False
                )
                summary2 = exp2.run()
                successful_runs.append((model_name, "context_scaling_gsm8k", exp2.paths["run_dir"]))
                print(f"[SUCCESS] Experiment 2 completed -> {exp2.paths['run_dir']}")
            except Exception as e:
                print(f"[ERROR] Experiment 2 failed for {model_name}: {e}")
                failed_runs.append((model_name, "context_scaling_gsm8k", str(e)))

        # 3. Experiment 3: BM25 RAG Energy Decomposition
        if "rag" in experiments_to_execute:
            print(f"\n--- [3/3] Running Experiment 3: BM25 RAG Pipeline ({model_name}) ---")
            rag_args = dict(base_cli_args)
            rag_args["top_k_list"] = args.top_k_list
            rag_args["include_baseline"] = True
            try:
                exp3 = RAGExperiment(
                    cli_args=rag_args,
                    interactive=False
                )
                summary3 = exp3.run()
                successful_runs.append((model_name, "rag_gsm8k", exp3.paths["run_dir"]))
                print(f"[SUCCESS] Experiment 3 completed -> {exp3.paths['run_dir']}")
            except Exception as e:
                print(f"[ERROR] Experiment 3 failed for {model_name}: {e}")
                failed_runs.append((model_name, "rag_gsm8k", str(e)))

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"Total Successful Runs: {len(successful_runs)}")
    for model, exp_type, run_path in successful_runs:
        print(f"  [OK] {model:20s} | {exp_type:24s} | {run_path}")

    if failed_runs:
        print(f"\nTotal Failed Runs: {len(failed_runs)}")
        for model, exp_type, err in failed_runs:
            print(f"  [FAILED] {model:20s} | {exp_type:24s} | Error: {err}")

    print("=" * 70)
    print("To generate publication figures, run:")
    print("  python scripts/generate_publication_figures.py --input-dir results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
