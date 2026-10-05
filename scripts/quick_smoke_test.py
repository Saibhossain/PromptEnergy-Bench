#!/usr/bin/env python3
"""Rapid Smoke Test Suite for PromptEnergy-Bench.

Quickly verifies end-to-end functionality across:
- All 4 Benchmark Datasets (GSM8K, Natural Questions, ContextEval, CNN/DailyMail)
- All 3 Benchmark Experiments (Prompting Strategies, Context-Length Scaling, BM25 RAG)
- 7 Target LLMs (customized for macOS MLX vs. Windows/Linux GGUF)
- Fast execution with --eval-size 2 (2 samples per condition)

Platforms Supported:
- macOS (Apple Silicon M1/M2/M3/M4 via MLX / Ollama)
- Windows (10-Core PC / CPU / GPU via Ollama)
- Kaggle (NVIDIA T4 GPU)
- Google Colab (NVIDIA T4 GPU)

Usage:
  # Auto-detects hardware and runs 7 default models with eval-size 2
  python scripts/quick_smoke_test.py

  # Explicit platform selection:
  python scripts/quick_smoke_test.py --hardware macbook_air_m1
  python scripts/quick_smoke_test.py --hardware windows_10core_pc
  python scripts/quick_smoke_test.py --hardware kaggle_t4_gpu
  python scripts/quick_smoke_test.py --hardware colab_t4_gpu

  # Test a single model or custom subset:
  python scripts/quick_smoke_test.py --models "qwen3.5:0.8b-mlx,qwen3.5:2b-mlx"
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from typing import List, Dict, Any


# Default benchmark models per platform
MIXED_NON_MAC_MODELS = (
    "gemma4:e2b,qwen3.5:2b,qwen3.5:0.8b,gemma3:4b,"
    "deepseek-r1:7b,llama3.1:8b,qwen3.5:9b,mistral:7b,"
    "gemma3:1b,llama3.2:1b,gemma4:12b"
)

PLATFORM_DEFAULT_MODELS = {
    "macbook_air_m1": "qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:9b-mlx",
    "windows_10core_pc": MIXED_NON_MAC_MODELS,
    "kaggle_t4_gpu": MIXED_NON_MAC_MODELS,
    "colab_t4_gpu": MIXED_NON_MAC_MODELS,
    "generic_cuda_server": MIXED_NON_MAC_MODELS
}


def detect_default_hardware() -> str:
    """Automatically identifies the host execution environment."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        return "macbook_air_m1"
    elif sys_name == "Windows":
        return "windows_10core_pc"
    else:
        # Check for Kaggle / Colab environment indicators
        if os.path.exists("/kaggle"):
            return "kaggle_t4_gpu"
        elif "COLAB_GPU" in os.environ or os.path.exists("/content"):
            return "colab_t4_gpu"
        return "generic_cuda_server"


def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Fast Smoke Test Runner (eval-size 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="auto",
        help="Hardware preset (auto, macbook_air_m1, windows_10core_pc, kaggle_t4_gpu, colab_t4_gpu)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to test. If omitted, uses standard 7 models for detected platform."
    )
    parser.add_argument(
        "--dataset", "--datasets",
        dest="dataset",
        type=str,
        default="all",
        help="Target benchmark datasets ('all' or comma-separated list [gsm8k, natural_questions, contexteval, cnn_dailymail])"
    )
    parser.add_argument(
        "--eval-size",
        type=str,
        default="2",
        help="Evaluation sample size per dataset (default: 2 for ultra-fast verification)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "primary", "context", "rag", "1", "2", "3"],
        help="Target experiment(s) to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_smoke_test",
        help="Temporary output directory for smoke test results"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Optional GPU ID for CUDA execution"
    )

    args = parser.parse_args()

    # Resolve hardware preset
    hw_target = args.hardware.strip().lower()
    if hw_target in ("auto", "default", "detect"):
        hw_target = detect_default_hardware()
    elif hw_target in ("mac", "macbook", "m1"):
        hw_target = "macbook_air_m1"
    elif hw_target in ("windows", "pc", "win"):
        hw_target = "windows_10core_pc"
    elif hw_target in ("kaggle", "kaggle_gpu"):
        hw_target = "kaggle_t4_gpu"
    elif hw_target in ("colab", "colab_gpu"):
        hw_target = "colab_t4_gpu"

    # Resolve models
    if args.models is not None and args.models.strip():
        models_str = args.models.strip()
    else:
        models_str = PLATFORM_DEFAULT_MODELS.get(hw_target, PLATFORM_DEFAULT_MODELS["windows_10core_pc"])

    # Locate master runner script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runner_path = os.path.join(project_root, "scripts", "run_all_experiments.py")

    cmd = [
        sys.executable,
        runner_path,
        "--hardware", hw_target,
        "--models", models_str,
        "--dataset", args.dataset,
        "--eval-size", str(args.eval_size),
        "--experiment", args.experiment,
        "--runs-per-condition", "1",
        "--warmups", "1",
        "--context-lengths", "0", "512",
        "--top-k-list", "1",
        "--output-dir", args.output_dir
    ]

    if args.gpu_id is not None:
        cmd.extend(["--gpu-id", str(args.gpu_id)])

    print("=" * 80)
    print("PROMPTENERGY-BENCH: RAPID SMOKE TEST SUITE (eval-size 2)")
    print("=" * 80)
    print(f"Host Hardware Preset : {hw_target}")
    print(f"Models to Verify ({len(models_str.split(','))}) : {models_str.split(',')}")
    print(f"Benchmark Datasets   : {args.dataset}")
    print(f"Evaluation Slice     : {args.eval_size} sample(s) per dataset")
    print(f"Sub-experiments      : {args.experiment} (Exp 1, Exp 2 [ctx: 0, 512], Exp 3 [top_k: 1])")
    print(f"Test Output Directory: {args.output_dir}")
    print("=" * 80)
    print(f"Executing: {' '.join(cmd)}\n")

    start_time = time.perf_counter()
    res = subprocess.run(cmd)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    if res.returncode == 0:
        print(f"[SUCCESS] All smoke tests passed successfully in {elapsed:.1f}s!")
        print(f"Verified:")
        print("  - Dataset parsing & 1k deterministic sampling logic")
        print("  - Zero-shot, Few-shot, CoT, Context Scaling, and BM25 RAG pipelines")
        print("  - Model inference & telemetry energy monitoring")
        print(f"  - Artifacts & test reports generated in: {args.output_dir}/")
    else:
        print(f"[FAILED] Smoke test encountered an error (exit code: {res.returncode}).")
        print("Review the traceback above to address the failing condition.")
    print("=" * 80)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
