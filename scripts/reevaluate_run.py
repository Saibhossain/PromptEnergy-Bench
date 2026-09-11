"""Re-evaluates a run directory with corrected evaluation and metrics rules.

Reads `results.jsonl`, re-evaluates all records using GSM8KEvaluator with strict
truncation invalidation, computes all metrics, updates summary.json, and re-generates
all publication tables and figures.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.evaluation.metrics import compute_experiment_metrics
from src.analysis.tables import generate_all_tables
from visual.plot_results import generate_all_plots_for_run


def reevaluate_run(run_dir: str):
    results_path = os.path.join(run_dir, "results.jsonl")
    metadata_path = os.path.join(run_dir, "metadata.json")
    config_path = os.path.join(run_dir, "config.json")
    summary_path = os.path.join(run_dir, "summary.json")
    tables_dir = os.path.join(run_dir, "tables")

    if not os.path.exists(results_path):
        print(f"Error: {results_path} does not exist.")
        return

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Read and re-evaluate each record
    re_evaluated_records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            
            # Check truncation
            is_truncated = bool(
                rec.get("generation_truncated") is True or
                rec.get("generation_stop_reason") == "length"
            )
            
            raw_output = rec.get("raw_output") or rec.get("text", "")
            raw_response = rec.get("raw_response", "")
            gold_answer = rec.get("gold_answer", "")

            eval_res = GSM8KEvaluator.evaluate(
                raw_output=raw_output,
                gold_answer=gold_answer,
                raw_response=raw_response,
                generation_truncated=is_truncated
            )

            rec["generation_truncated"] = is_truncated
            rec["extracted_answer"] = eval_res.extracted_answer
            rec["answer_parse_success"] = eval_res.answer_parse_success
            rec["answer_correct"] = eval_res.answer_correct
            rec["exact_match"] = eval_res.exact_match
            if is_truncated:
                rec["error_type"] = "generation_truncated"
            else:
                rec["error_type"] = None
            
            re_evaluated_records.append(rec)

    # Re-write updated results.jsonl safely
    with open(results_path, "w", encoding="utf-8") as f:
        for rec in re_evaluated_records:
            f.write(json.dumps(rec) + "\n")

    # Recompute metrics
    metrics = compute_experiment_metrics(re_evaluated_records)

    summary = {
        "run_id": metadata.get("run_id", os.path.basename(run_dir)),
        "experiment_name": metadata.get("experiment_name", "primary_exp_gsm8k"),
        "timestamp": metadata.get("timestamp"),
        "device": metadata.get("device", {}).get("name"),
        "model": metadata.get("backend", {}).get("model_name"),
        "evaluation_size": config.get("dataset", {}).get("evaluation_size", 50),
        "metrics": metrics
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Re-generate tables
    generate_all_tables(
        records=re_evaluated_records,
        metadata=metadata,
        config=config,
        summary=summary,
        tables_dir=tables_dir
    )

    # Re-generate plots
    generate_all_plots_for_run(
        target_dir=run_dir,
        results_file=results_path
    )

    print(f"Successfully re-evaluated {len(re_evaluated_records)} records in {run_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()
    reevaluate_run(args.run_dir)
