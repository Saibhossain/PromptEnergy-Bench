#!/usr/bin/env python3
"""Pipeline Validation & Smoke Test Suite for PromptEnergy-Bench.

Executes a complete, end-to-end verification run across:
- All 4 Benchmark Datasets: gsm8k, natural_questions, contexteval, cnn_dailymail
- All 3 Benchmark Experiments: Prompting Strategies (Exp 1), Context Scaling (Exp 2), BM25 RAG (Exp 3)
- 3 Target Models: qwen3.5:2b, qwen3.5:0.8b, gemma3:4b
- Evaluation Sample Size: 10 samples per condition
- Hardware: windows_10core_pc (or windows_cuda_pc if GPU is available)

Usage:
  python scripts/test_pipeline_run.py
  python scripts/test_pipeline_run.py --hardware windows_cuda_pc
  python scripts/test_pipeline_run.py --eval-size 5
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Pipeline Verification & Smoke Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="windows_10core_pc",
        help="Hardware preset (e.g., windows_10core_pc, windows_cuda_pc, macbook_air_m1)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3.5:2b,qwen3.5:0.8b,gemma3:4b",
        help="Comma-separated model identifiers"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset(s) to evaluate: 'all' or comma-separated list [gsm8k, natural_questions, contexteval, cnn_dailymail]"
    )
    parser.add_argument(
        "--eval-size",
        type=str,
        default="10",
        help="Evaluation sample size per dataset"
    )
    parser.add_argument(
        "--runs-per-condition",
        type=int,
        default=1,
        help="Repetitions per condition"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "primary", "context", "rag", "1", "2", "3"],
        help="Experiment module(s) to execute"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip completed condition checkpoints"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running all conditions even if checkpoints exist"
    )

    args = parser.parse_args()

    # Determine Python executable
    py_exec = sys.executable

    runner_script = os.path.join(os.path.dirname(__file__), "run_all_experiments.py")

    cmd = [
        py_exec,
        runner_script,
        "--hardware", args.hardware,
        "--models", args.models,
        "--dataset", args.dataset,
        "--experiment", args.experiment,
        "--eval-size", str(args.eval_size),
        "--runs-per-condition", str(args.runs_per_condition)
    ]

    if args.skip_existing and not args.force_rerun:
        cmd.append("--skip-existing")

    print("=" * 75)
    print("PROMPTENERGY-BENCH: PIPELINE VALIDATION & SMOKE TEST RUNNER")
    print("=" * 75)
    print(f"Hardware Target       : {args.hardware}")
    print(f"Models Selected       : {args.models}")
    print(f"Datasets Selected     : {args.dataset}")
    print(f"Evaluation Size       : {args.eval_size} samples/dataset")
    print(f"Runs Per Condition    : {args.runs_per_condition}")
    print(f"Target Experiment(s)  : {args.experiment}")
    print(f"Executing Script      : {runner_script}")
    print("=" * 75)
    print(f"\nCommand: {' '.join(cmd)}\n")

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n" + "=" * 75)
        print("[SUCCESS] All pipeline smoke tests finished successfully!")
        print("Deliverables generated under: results/")
        print("Master comparison report    : results/master_comparison/")
        print("=" * 75)
    else:
        print("\n" + "=" * 75)
        print(f"[ERROR] Pipeline test failed with exit code: {res.returncode}")
        print("=" * 75)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
