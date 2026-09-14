#!/usr/bin/env python3
"""CLI Launcher for Experiment 2: Controlled Context Scaling on GSM8K."""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.context_scaling import ContextScalingExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Controlled Context Scaling on GSM8K (PromptEnergy-Bench)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Device flags
    parser.add_argument("--device-name", type=str, default=None, help="Device name")
    parser.add_argument("--os", type=str, default=None, help="Operating system")
    parser.add_argument("--cpu", type=str, default=None, help="CPU model")
    parser.add_argument("--gpu", type=str, default=None, help="GPU available ('yes' or 'no')")
    parser.add_argument("--gpu-count", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--gpu-name", type=str, default=None, help="GPU model name")
    parser.add_argument("--gpu-vram-gb", type=float, default=None, help="GPU VRAM in GB")
    parser.add_argument("--ram-gb", type=int, default=None, help="System RAM in GB")

    # Backend / model flags
    parser.add_argument("--operator", type=str, default=None, help="Model operator (ollama, openai, transformers, mlx, llama.cpp)")
    parser.add_argument("--model-format", type=str, default=None, help="Model format (mlx, gguf, f16, bf16, fp8, int8, api)")
    parser.add_argument("--model", type=str, default=None, help="Model name")

    # Context experiment specific
    parser.add_argument("--context-lengths", nargs="+", type=int, default=None, help="List of context token lengths (e.g. 0 512 1024 2048 4096)")
    parser.add_argument("--include-8k", action="store_true", help="Attempt 8192 token context scaling if model permits")
    parser.add_argument("--context-type", type=str, default="relevant", choices=["relevant", "distractor"], help="Context type ('relevant' or 'distractor')")

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

    is_interactive = not args.non_interactive and not args.validation and args.resume is None and args.model is None

    if args.validation:
        cli_dict["eval_size"] = 50
        if cli_dict.get("warmups") is None:
            cli_dict["warmups"] = 1
        if cli_dict.get("repetitions") is None:
            cli_dict["repetitions"] = 1

    exp = ContextScalingExperiment(cli_args=cli_dict, interactive=is_interactive)
    summary = exp.run()

    if args.validation or exp.eval_size == 50 or isinstance(exp.eval_size, int):
        metrics = summary.get("metrics", {})
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY: CONTROLLED CONTEXT SCALING")
        print("=" * 60)
        print(f"Dataset: GSM8K")
        print(f"Evaluation split: TEST")
        print(f"Evaluation examples: {summary.get('evaluation_size', exp.eval_size)}")
        print(f"Context type: {exp.context_type}")
        print(f"Context lengths: {exp.context_lengths}")
        print(f"Model: {summary.get('model')}")
        print(f"Backend: {summary.get('backend')}")
        print(f"Device: {summary.get('device')}")
        print(f"Accuracy: {metrics.get('accuracy', 0.0)}")
        print(f"Mean TTFT: {metrics.get('mean_ttft_ms', 'N/A')} ms")
        print(f"Mean latency: {metrics.get('mean_total_latency_ms', metrics.get('mean_latency_ms', 'N/A'))} ms")
        print(f"Mean CPU: {metrics.get('mean_cpu_percent', 'N/A')} % (Peak: {metrics.get('peak_cpu_percent', 'N/A')} %)")
        print(f"Mean RAM: {metrics.get('mean_ram_used_gb', 'N/A')} GB")
        print(f"Mean energy: {metrics.get('mean_energy_j', 'N/A')} J")
        print("=" * 60)
        print(f"Results saved to: {exp.paths['run_dir']}\n")


if __name__ == "__main__":
    main()
