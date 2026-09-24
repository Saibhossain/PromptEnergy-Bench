#!/usr/bin/env python3
"""Dataset Downloader for PromptEnergy-Bench.

Downloads and pre-processes benchmark datasets from HuggingFace Hub:
1. Mathematical Reasoning : openai/gsm8k (config: 'main')
2. Knowledge-Intensive QA : sentence-transformers/natural-questions (config: 'pair')
3. Long-Context QA        : allenai/ContextEval (config: 'main')
4. Summarization          : abisee/cnn_dailymail (config: '3.0.0')

Saves both HuggingFace Arrow format (via save_to_disk) and line-delimited JSONL files.

Usage:
  python scripts/download_datasets.py --datasets all
  python scripts/download_datasets.py --datasets gsm8k natural_questions cnn_dailymail
  python scripts/download_datasets.py --datasets cnn_dailymail --output-dir custom_datasets/
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package is required. Install with: pip install datasets")
    sys.exit(1)


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "config": "main",
        "folder": "gsm8k",
        "description": "Mathematical Reasoning (GSM8K Grade School Math)",
        "default_splits": ["train", "test"]
    },
    "natural_questions": {
        "hf_path": "sentence-transformers/natural-questions",
        "config": "pair",
        "folder": "natural_questions",
        "description": "Knowledge-Intensive QA (Natural Questions)",
        "default_splits": ["train"]
    },
    "contexteval": {
        "hf_path": "allenai/ContextEval",
        "config": "main",
        "folder": "contexteval",
        "description": "Long-Context Evaluation Benchmark (ContextEval)",
        "default_splits": ["test"]
    },
    "cnn_dailymail": {
        "hf_path": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "folder": "cnn_dailymail",
        "description": "Abstractive Summarization (CNN / DailyMail 3.0.0)",
        "default_splits": ["train", "validation", "test"]
    }
}

# Aliases for flexible CLI specification
ALIASES = {
    "nq": "natural_questions",
    "natural-questions": "natural_questions",
    "sentence-transformers/natural-questions": "natural_questions",
    "context_eval": "contexteval",
    "context-eval": "contexteval",
    "allenai/contexteval": "contexteval",
    "allenai/ContextEval": "contexteval",
    "cnn": "cnn_dailymail",
    "cnn-dailymail": "cnn_dailymail",
    "abisee/cnn_dailymail": "cnn_dailymail",
    "openai/gsm8k": "gsm8k"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and export benchmark datasets for PromptEnergy-Bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names to download ('all', 'gsm8k', 'natural_questions', 'contexteval', 'cnn_dailymail') or HF path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Root directory where downloaded datasets will be stored"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples per split to export to JSONL (useful for quick testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing dataset files"
    )
    return parser.parse_args()


def resolve_dataset_keys(requested: List[str]) -> List[str]:
    """Resolves CLI argument tokens to canonical DATASET_REGISTRY keys."""
    if "all" in [r.lower() for r in requested]:
        return list(DATASET_REGISTRY.keys())

    resolved = []
    for item in requested:
        cleaned = item.strip().lower()
        if cleaned in DATASET_REGISTRY:
            resolved.append(cleaned)
        elif cleaned in ALIASES:
            resolved.append(ALIASES[cleaned])
        else:
            # Check matching against hf_path
            matched = False
            for k, meta in DATASET_REGISTRY.items():
                if item.strip().lower() == meta["hf_path"].lower():
                    resolved.append(k)
                    matched = True
                    break
            if not matched:
                print(f"[WARN] Unknown dataset '{item}'. Available: {list(DATASET_REGISTRY.keys())}")
    return list(dict.fromkeys(resolved))  # preserve order & deduplicate


def download_single_dataset(
    key: str,
    output_root: str,
    max_samples: Optional[int] = None,
    force: bool = False
) -> bool:
    """Downloads a single dataset and exports both Arrow disk format and JSONL splits."""
    meta = DATASET_REGISTRY[key]
    dest_dir = os.path.join(output_root, meta["folder"])
    os.makedirs(dest_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"Dataset   : {meta['description']}")
    print(f"HF Path   : {meta['hf_path']} (config: {meta['config']})")
    print(f"Target Dir: {dest_dir}")
    print("=" * 65)

    # Check if JSONL already exists
    already_exists = True
    for split in meta["default_splits"]:
        jsonl_path = os.path.join(dest_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
            already_exists = False
            break

    if already_exists and not force:
        print(f"[SKIP] Dataset '{key}' already exists in {dest_dir}. Use --force to re-download.")
        return True

    t0 = time.time()
    try:
        print(f"Fetching '{meta['hf_path']}' from HuggingFace...")
        if meta["config"]:
            dataset = load_dataset(meta["hf_path"], meta["config"])
        else:
            dataset = load_dataset(meta["hf_path"])
        
        # Save to disk in Arrow format
        try:
            arrow_dir = os.path.join(dest_dir, "hf_arrow")
            dataset.save_to_disk(arrow_dir)
            print(f"  [SAVED] Arrow dataset -> {arrow_dir}")
        except Exception as e:
            print(f"  [INFO] Arrow save_to_disk note: {e}")

        # Export each split as line-delimited JSONL
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            jsonl_file = os.path.join(dest_dir, f"{split_name}.jsonl")
            
            if max_samples is not None and max_samples > 0:
                split_data = split_data.select(range(min(max_samples, len(split_data))))

            print(f"  [EXPORT] Exporting split '{split_name}' ({len(split_data):,} rows) -> {jsonl_file}...")
            split_data.to_json(jsonl_file, orient="records", lines=True)

        elapsed = time.time() - t0
        print(f"[SUCCESS] {meta['description']} downloaded & exported in {elapsed:.2f}s.")
        return True

    except Exception as e:
        print(f"[ERROR] Failed downloading {meta['hf_path']}: {e}")
        return False


def main():
    args = parse_args()
    keys_to_download = resolve_dataset_keys(args.datasets)

    if not keys_to_download:
        print("No valid datasets specified for download. Exiting.")
        sys.exit(1)

    print("=" * 65)
    print("PROMPTENERGY-BENCH: DATASET DOWNLOAD SUITE")
    print("=" * 65)
    print(f"Selected Datasets: {keys_to_download}")
    print(f"Output Directory : {args.output_dir}")
    if args.max_samples:
        print(f"Max Samples/Split: {args.max_samples}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for key in keys_to_download:
        ok = download_single_dataset(
            key=key,
            output_root=args.output_dir,
            max_samples=args.max_samples,
            force=args.force
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"Successfully processed : {success_count}/{len(keys_to_download)}")
    if fail_count > 0:
        print(f"Failed downloads       : {fail_count}/{len(keys_to_download)}")
    print(f"Datasets directory     : {os.path.abspath(args.output_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()