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
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Default maximum generated tokens override")
    parser.add_argument("--generation-config", type=str, default=None, help="Path to generation YAML configuration")

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
        gen_cfg = exp.config.get("generation", {})
        default_max = gen_cfg.get("default_max_tokens", "N/A")
        strat_max = gen_cfg.get("strategy_max_tokens", {})
        strat_max_str = ", ".join(f"{k}: {v}" for k, v in strat_max.items()) if strat_max else "N/A"

        trunc_count = metrics.get("truncated_generations", 0)
        trunc_rate = metrics.get("truncation_rate", 0.0)
        pass_status = "PASS" if metrics.get("failed_inference_runs", 0) == 0 else "FAIL"

        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION SUMMARY")
        print("=" * 60)
        print("Dataset: GSM8K")
        print("Evaluation split: TEST")
        print(f"Evaluation examples: {summary.get('evaluation_examples', 50)}")
        print(f"Strategies: {summary.get('strategies', 5)}")
        print(f"Inference runs: {metrics.get('requested_inference_runs', 250)}")
        print(f"\nModel: {summary.get('model')}")
        print(f"Backend: {summary.get('backend')}")
        print(f"Device: {summary.get('device')}")
        print(f"\nGeneration configuration:")
        print(f"Default max tokens: {default_max}")
        print(f"Strategy-specific max tokens: {strat_max_str}")
        print(f"\nTruncated generations: {trunc_count}")
        print(f"Truncation rate: {trunc_rate * 100.0:.2f}%")
        print(f"\nAccuracy: {metrics.get('accuracy', 0.0) * 100.0:.2f}%")
        print(f"Mean energy: {metrics.get('mean_energy_j')} J")
        print(f"Mean latency: {metrics.get('mean_latency_ms')} ms")
        print(f"Mean thinking tokens: {metrics.get('mean_thinking_tokens')}")
        print(f"\nPipeline status: {pass_status}")
        print("=" * 60)
        print("\nValidation completed.")
        print("Full research benchmark was NOT executed.")
        print(f"Results saved to: {exp.paths['run_dir']}\n")


if __name__ == "__main__":
    main()
