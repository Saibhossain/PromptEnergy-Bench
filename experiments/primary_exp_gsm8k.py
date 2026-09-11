#!/usr/bin/env python3
"""CLI Launcher for Experiment 1: Standard Prompting vs Few-Shot vs Chain-of-Thought on GSM8K."""

import argparse
import sys
import os

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.primary import PrimaryExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Prompting Strategies on GSM8K (PromptEnergy-Bench)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Device flags
    parser.add_argument("--device-name", type=str, default=None, help="Device name (e.g. 'MacBook Air M1')")
    parser.add_argument("--os", type=str, default=None, help="Operating system (macOS, Linux, Windows)")
    parser.add_argument("--cpu", type=str, default=None, help="CPU model")
    parser.add_argument("--gpu", type=str, default=None, help="GPU available ('yes' or 'no')")
    parser.add_argument("--gpu-count", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--gpu-name", type=str, default=None, help="GPU model name")
    parser.add_argument("--gpu-vram-gb", type=float, default=None, help="GPU VRAM in GB")
    parser.add_argument("--ram-gb", type=int, default=None, help="System RAM in GB")

    # Backend / model flags
    parser.add_argument("--operator", type=str, default=None, help="Model operator (ollama, openai, transformers, mlx, llama.cpp)")
    parser.add_argument("--model-format", type=str, default=None, help="Model format (mlx, gguf, f16, bf16, fp8, int8, api)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. 'qwen3.5:0.8b-mlx')")

    # Dataset / execution flags
    parser.add_argument("--eval-size", type=str, default=None, help="Evaluation dataset size ('50' or 'full')")
    parser.add_argument("--warmups", type=int, default=None, help="Number of warmup runs")
    parser.add_argument("--repetitions", type=int, default=None, help="Number of repetitions per condition")
    parser.add_argument("--energy-mode", type=str, default=None, help="Energy measurement mode")

    # Sampling flags
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum generated tokens")

    # Modes
    parser.add_argument("--validation", action="store_true", help="Enable validation mode (50 examples, 1 warmup, 1 rep)")
    parser.add_argument("--resume", type=str, default=None, help="Resume an existing experiment from its directory")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt interactively, use defaults or flags")

    return parser.parse_args()


def main():
    args = parse_args()
    cli_dict = vars(args)

    # Determine interactive mode
    is_interactive = not args.non_interactive and not args.validation and args.resume is None and args.model is None

    # Enforce validation defaults if flag set
    if args.validation:
        cli_dict["eval_size"] = 50
        if cli_dict.get("warmups") is None:
            cli_dict["warmups"] = 1
        if cli_dict.get("repetitions") is None:
            cli_dict["repetitions"] = 1

    exp = PrimaryExperiment(cli_args=cli_dict, interactive=is_interactive)
    summary = exp.run()

    # If validation mode was run, output the exact required validation summary format
    if args.validation or exp.eval_size == 50:
        metrics = summary.get("metrics", {})
        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Dataset: GSM8K")
        print(f"Evaluation split: TEST")
        print(f"Evaluation examples: {summary.get('evaluation_size', 50)}")
        print(f"Model: {summary.get('model')}")
        print(f"Backend: {summary.get('backend')}")
        print(f"Device: {summary.get('device')}")
        print(f"Strategies: {summary.get('metrics', {}).get('requested_samples', 0)} total requests")
        print(f"Correct: {metrics.get('correct_answers', 0)}")
        print(f"Accuracy: {metrics.get('accuracy', 0.0)}")
        print(f"Mean latency: {metrics.get('mean_latency_ms')} ms")
        print(f"Mean output tokens: {metrics.get('mean_output_tokens')}")
        print(f"Mean energy: {metrics.get('mean_energy_j')} J")
        print(f"Energy measurement: {exp.energy_mode}")
        print(f"Errors: {metrics.get('failed_samples', 0)}")
        print("=" * 60)
        print("Pipeline status: PASS" if metrics.get("failed_samples", 0) == 0 else "Pipeline status: COMPLETED WITH ERRORS")
        print("\nValidation completed. Full research benchmark was NOT executed.")
        print(f"Results saved to: {exp.paths['run_dir']}\n")


if __name__ == "__main__":
    main()
