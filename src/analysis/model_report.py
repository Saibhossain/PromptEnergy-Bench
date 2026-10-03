"""Comprehensive Single-Model Analytics & Report Generator for PromptEnergy-Bench.

Generates aggregated publication tables and Seaborn visualizations for a single LLM
after it completes all scheduled experiments (Prompting, Context Scaling, RAG)
across all datasets.

Outputs inside the model experiment folder (e.g. results/windows_10core_pc/exp_1_qwen2B_10/):
- tables/
    - table1_full_merged_results.csv / .md / .tex
    - table2_model_execution_energy.csv / .md / .tex
    - table3_strategy_summary.csv / .md
    - table4_dataset_summary.csv / .md
    - table5_pareto_frontier.csv / .md
    - table6_meg_efficiency.csv / .md
- plots/
    - 1_model_accuracy_by_dataset_and_strategy.png / .pdf
    - 2_latency_by_strategy_and_context.png / .pdf
    - 3_energy_vs_accuracy_pareto_tradeoff.png / .pdf
    - 4_energy_breakdown_prefill_decode.png / .pdf
    - 5_rag_energy_and_accuracy_scaling.png / .pdf
    - 6_model_runtime_token_throughput.png / .pdf
    - 7_comprehensive_model_dashboard.png / .pdf
- summary_all_experiments.json
"""

import csv
import glob
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.pareto import compute_pareto_frontier


# ============================================================
# Model Tag Cleaning Helper
# ============================================================

def get_clean_model_tag(model_name: str) -> str:
    """Produces clean, compact model tag (e.g. 'qwen08B', 'qwen2B', 'gemma4B')."""
    m = str(model_name).lower().replace("-mlx", "").replace(":latest", "").strip()
    if "qwen" in m:
        if "0.8b" in m or "0.8" in m:
            return "qwen08B"
        elif "0.5b" in m or "0.5" in m:
            return "qwen05B"
        elif "2b" in m or "2.0b" in m:
            return "qwen2B"
        elif "3b" in m or "3.0b" in m:
            return "qwen3B"
        elif "7b" in m:
            return "qwen7B"
        elif "14b" in m:
            return "qwen14B"
    if "gemma" in m:
        if "4b" in m:
            return "gemma4B"
        elif "2b" in m:
            return "gemma2B"
        elif "9b" in m:
            return "gemma9B"
    if "llama" in m:
        if "1b" in m:
            return "llama1B"
        elif "3b" in m:
            return "llama3B"
        elif "8b" in m:
            return "llama8B"
    
    # Generic fallback: alphanumeric only
    clean = re.sub(r"[^a-zA-Z0-9]", "", m)
    return clean or "model"


# ============================================================
# Strategy Display & Sorting Definitions
# ============================================================

STRATEGY_DISPLAY = {
    "zero_shot_direct": "Zero-shot Direct",
    "few_shot_3": "Few-shot (3)",
    "zero_shot_cot": "Zero-shot CoT",
    "short_cot": "Short CoT",
    "long_cot": "Long CoT",
    "ctx_0": "Context 0 tokens",
    "ctx_512": "Context 512 tokens",
    "ctx_1024": "Context 1024 tokens",
    "ctx_2048": "Context 2048 tokens",
    "ctx_4096": "Context 4096 tokens",
    "ctx_8192": "Context 8192 tokens",
    "rag_top_1": "BM25 RAG (top-1)",
    "rag_top_3": "BM25 RAG (top-3)",
    "rag_top_5": "BM25 RAG (top-5)"
}

DATASET_DISPLAY = {
    "gsm8k": "GSM8K (Math)",
    "natural_questions": "Natural Questions (QA)",
    "contexteval": "ContextEval (Long QA)",
    "cnn_dailymail": "CNN/DailyMail (Summary)"
}


def _format_num(val: Any, decimals: int = 2, fallback: str = "N/A") -> str:
    """Formats numeric values safely."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        return f"{val:.{decimals}f}"
    return str(val)


# ============================================================
# Ingestion & Aggregation
# ============================================================

def collect_model_records(model_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Walks through model experiment folder and ingests all results.jsonl records."""
    all_records = []
    meta_info = {}

    for root, dirs, files in os.walk(model_dir):
        if "plots" in root or "tables" in root:
            continue
        for f in files:
            if f in ("results.jsonl", "raw_results.jsonl"):
                fpath = os.path.join(root, f)
                # Check for metadata in same folder
                meta_path = os.path.join(root, "metadata.json")
                if os.path.exists(meta_path) and not meta_info:
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta_info = json.load(mf)
                    except Exception:
                        pass

                try:
                    with open(fpath, "r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                # Tag source folder
                                rec["_source_dir"] = os.path.basename(root)
                                all_records.append(rec)
                            except json.JSONDecodeError:
                                pass
                except Exception as e:
                    print(f"Warning: error reading {fpath}: {e}")

    # Deduplicate records by condition_key or (sample_id, strategy, dataset, repetition)
    unique_records = []
    seen = set()
    for r in all_records:
        key = r.get("condition_key") or f"{r.get('dataset')}_{r.get('strategy')}_{r.get('sample_id')}_{r.get('repetition', 1)}"
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    return unique_records, meta_info


# ============================================================
# Table Generation Functions
# ============================================================

def generate_table1_full_merged(records: List[Dict[str, Any]], output_dir: str, model_name: str) -> pd.DataFrame:
    """Generates Table 1: Full Merged Results across all datasets, experiments, and strategies."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure required columns exist
    if "dataset" not in df.columns:
        df["dataset"] = df.get("task_type", "unknown")
    if "strategy" not in df.columns:
        df["strategy"] = "unknown"

    # Extract numerical fields
    def get_val(r, *keys, default=None):
        for k in keys:
            if k in r and r[k] is not None:
                return r[k]
        return default

    rows = []
    # Group by (dataset, strategy)
    grouped = df.groupby(["dataset", "strategy"], sort=False)

    for (ds, strat), g in grouped:
        n_samples = len(g)
        
        # Accuracy & correctness
        correct_list = [r for r in g["answer_correct"] if r is not None]
        acc_pct = (sum(bool(x) for x in correct_list) / len(correct_list) * 100.0) if correct_list else None

        # F1 / Span match
        f1_vals = [r.get("metric_values", {}).get("token_f1") for _, r in g.iterrows() if isinstance(r.get("metric_values"), dict) and r.get("metric_values", {}).get("token_f1") is not None]
        mean_f1 = (sum(f1_vals) / len(f1_vals)) if f1_vals else None

        # ROUGE-L
        rouge_vals = [r.get("metric_values", {}).get("rougeL_f1") for _, r in g.iterrows() if isinstance(r.get("metric_values"), dict) and r.get("metric_values", {}).get("rougeL_f1") is not None]
        mean_rouge = (sum(rouge_vals) / len(rouge_vals)) if rouge_vals else None

        # Energy metrics
        energy_total = [r.get("energy_total_j") for _, r in g.iterrows() if r.get("energy_total_j") is not None]
        energy_net = [r.get("energy_net_j") for _, r in g.iterrows() if r.get("energy_net_j") is not None]
        energy_prefill = [r.get("energy_prefill_j") for _, r in g.iterrows() if r.get("energy_prefill_j") is not None]
        energy_decode = [r.get("energy_decode_j") for _, r in g.iterrows() if r.get("energy_decode_j") is not None]

        mean_e_tot = (sum(energy_total) / len(energy_total)) if energy_total else None
        mean_e_net = (sum(energy_net) / len(energy_net)) if energy_net else None
        mean_e_pref = (sum(energy_prefill) / len(energy_prefill)) if energy_prefill else None
        mean_e_dec = (sum(energy_decode) / len(energy_decode)) if energy_decode else None

        # Latency metrics
        ttft_vals = [r.get("ttft_ms") for _, r in g.iterrows() if r.get("ttft_ms") is not None]
        tot_lat_vals = [r.get("total_latency_ms") for _, r in g.iterrows() if r.get("total_latency_ms") is not None]
        gen_lat_vals = [r.get("generation_latency_ms") for _, r in g.iterrows() if r.get("generation_latency_ms") is not None]

        mean_ttft = (sum(ttft_vals) / len(ttft_vals)) if ttft_vals else None
        mean_tot_lat = (sum(tot_lat_vals) / len(tot_lat_vals)) if tot_lat_vals else None
        mean_gen_lat = (sum(gen_lat_vals) / len(gen_lat_vals)) if gen_lat_vals else None

        # Token metrics
        in_toks = [r.get("input_tokens") for _, r in g.iterrows() if r.get("input_tokens") is not None]
        out_toks = [r.get("output_tokens") for _, r in g.iterrows() if r.get("output_tokens") is not None]
        think_toks = [r.get("thinking_tokens") for _, r in g.iterrows() if r.get("thinking_tokens") is not None]

        mean_in_tok = (sum(in_toks) / len(in_toks)) if in_toks else 0
        mean_out_tok = (sum(out_toks) / len(out_toks)) if out_toks else 0
        mean_think_tok = (sum(think_toks) / len(think_toks)) if think_toks else 0

        # Throughput & Unit Energy
        tok_per_sec = (mean_out_tok / (mean_gen_lat / 1000.0)) if (mean_gen_lat and mean_gen_lat > 0 and mean_out_tok > 0) else None
        mj_per_tok = ((mean_e_tot * 1000.0) / (mean_in_tok + mean_out_tok)) if (mean_e_tot and (mean_in_tok + mean_out_tok) > 0) else None

        # Truncation rate
        trunc_list = [bool(r.get("generation_truncated", False)) for _, r in g.iterrows()]
        trunc_rate = (sum(trunc_list) / len(trunc_list) * 100.0) if trunc_list else 0.0

        # Determine Experiment Type label
        exp_type = "Prompting (Exp 1)"
        if str(strat).startswith("ctx_"):
            exp_type = "Context Scaling (Exp 2)"
        elif str(strat).startswith("rag_"):
            exp_type = "BM25 RAG (Exp 3)"

        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "Experiment Type": exp_type,
            "Strategy": STRATEGY_DISPLAY.get(strat, strat),
            "Raw Strategy": strat,
            "Raw Dataset": ds,
            "Samples": n_samples,
            "Accuracy (%)": round(acc_pct, 2) if acc_pct is not None else None,
            "Token F1": round(mean_f1, 4) if mean_f1 is not None else None,
            "ROUGE-L": round(mean_rouge, 4) if mean_rouge is not None else None,
            "Total Energy (J)": round(mean_e_tot, 3) if mean_e_tot is not None else None,
            "Net Energy (J)": round(mean_e_net, 3) if mean_e_net is not None else None,
            "Prefill Energy (J)": round(mean_e_pref, 3) if mean_e_pref is not None else None,
            "Decode Energy (J)": round(mean_e_dec, 3) if mean_e_dec is not None else None,
            "TTFT (ms)": round(mean_ttft, 1) if mean_ttft is not None else None,
            "Decode Latency (ms)": round(mean_gen_lat, 1) if mean_gen_lat is not None else None,
            "Total Latency (ms)": round(mean_tot_lat, 1) if mean_tot_lat is not None else None,
            "Input Tokens": round(mean_in_tok, 1),
            "Output Tokens": round(mean_out_tok, 1),
            "Thinking Tokens": round(mean_think_tok, 1) if mean_think_tok > 0 else "N/A",
            "Throughput (tok/s)": round(tok_per_sec, 2) if tok_per_sec is not None else None,
            "Energy/Token (mJ)": round(mj_per_tok, 2) if mj_per_tok is not None else None,
            "Truncation Rate (%)": round(trunc_rate, 1)
        })

    summary_df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "table1_full_merged_results.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save Markdown
    md_path = os.path.join(output_dir, "table1_full_merged_results.md")
    display_cols = [c for c in summary_df.columns if not c.startswith("Raw ")]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Table 1: Full Merged Results for Model: {model_name}\n\n")
        f.write(summary_df[display_cols].to_markdown(index=False))
        f.write("\n")

    # Save LaTeX
    tex_path = os.path.join(output_dir, "table1_full_merged_results.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 1: Full Merged Results\n")
        f.write(summary_df[display_cols].to_latex(index=False, escape=True))
        f.write("\n")

    return summary_df


def generate_table2_execution_energy(records: List[Dict[str, Any]], meta: Dict[str, Any], output_dir: str, model_name: str, hw_name: str) -> pd.DataFrame:
    """Generates Table 2: Model total execution time, total tokens, and CodeCarbon/hardware energy."""
    if not records:
        return pd.DataFrame()

    total_records = len(records)
    total_in_tokens = sum(r.get("input_tokens", 0) or 0 for r in records)
    total_out_tokens = sum(r.get("output_tokens", 0) or 0 for r in records)
    total_tokens = total_in_tokens + total_out_tokens

    # Latency
    tot_latencies_s = [((r.get("total_latency_ms") or 0) / 1000.0) for r in records if r.get("total_latency_ms")]
    total_inference_time_s = sum(tot_latencies_s)

    # Energy
    total_energies_j = [r.get("energy_total_j") for r in records if r.get("energy_total_j") is not None]
    net_energies_j = [r.get("energy_net_j") for r in records if r.get("energy_net_j") is not None]

    sum_energy_j = sum(total_energies_j) if total_energies_j else 0.0
    sum_net_energy_j = sum(net_energies_j) if net_energies_j else 0.0
    sum_energy_kwh = sum_energy_j / 3.6e6

    # Measurement method
    meas_method = meta.get("measurement", {}).get("energy_method", "codecarbon_estimated")
    active_power_samples = [r.get("active_power_w") for r in records if r.get("active_power_w") is not None]
    mean_active_power_w = (sum(active_power_samples) / len(active_power_samples)) if active_power_samples else (sum_energy_j / total_inference_time_s if total_inference_time_s > 0 else 0.0)

    idle_powers = [r.get("idle_power_w") for r in records if r.get("idle_power_w") is not None]
    mean_idle_power_w = (sum(idle_powers) / len(idle_powers)) if idle_powers else None

    # Accuracy
    correct_list = [r.get("answer_correct") for r in records if r.get("answer_correct") is not None]
    overall_acc = (sum(bool(x) for x in correct_list) / len(correct_list) * 100.0) if correct_list else 0.0

    # Unit energy
    mj_per_tok = (sum_energy_j * 1000.0 / total_tokens) if total_tokens > 0 else 0.0
    j_per_query = (sum_energy_j / total_records) if total_records > 0 else 0.0

    data = [
        {"Metric Parameter": "Model Name", "Measured Value": model_name, "Unit": "string"},
        {"Metric Parameter": "Hardware Environment", "Measured Value": hw_name, "Unit": "string"},
        {"Metric Parameter": "Energy Telemetry Adapter", "Measured Value": meas_method, "Unit": "method"},
        {"Metric Parameter": "Total Evaluated Inferences", "Measured Value": f"{total_records:,}", "Unit": "queries"},
        {"Metric Parameter": "Total Inference Duration", "Measured Value": f"{total_inference_time_s:.2f}", "Unit": "seconds"},
        {"Metric Parameter": "Total Inference Duration (Minutes)", "Measured Value": f"{total_inference_time_s / 60.0:.2f}", "Unit": "minutes"},
        {"Metric Parameter": "Total Prompt (Input) Tokens", "Measured Value": f"{total_in_tokens:,}", "Unit": "tokens"},
        {"Metric Parameter": "Total Generated (Output) Tokens", "Measured Value": f"{total_out_tokens:,}", "Unit": "tokens"},
        {"Metric Parameter": "Total Token Budget Ingested + Emitted", "Measured Value": f"{total_tokens:,}", "Unit": "tokens"},
        {"Metric Parameter": "Mean Generation Speed", "Measured Value": f"{(total_out_tokens / total_inference_time_s):.2f}" if total_inference_time_s > 0 else "N/A", "Unit": "tokens/sec"},
        {"Metric Parameter": "Total Gross Energy Consumed", "Measured Value": f"{sum_energy_j:.2f}", "Unit": "Joules (J)"},
        {"Metric Parameter": "Total Net Energy Consumed (Excl. Idle)", "Measured Value": f"{sum_net_energy_j:.2f}", "Unit": "Joules (J)"},
        {"Metric Parameter": "Total Electrical Energy in kWh", "Measured Value": f"{sum_energy_kwh:.6f}", "Unit": "kWh"},
        {"Metric Parameter": "Mean Active Power Draw", "Measured Value": f"{mean_active_power_w:.2f}", "Unit": "Watts (W)"},
        {"Metric Parameter": "Calibrated Idle Power Baseline", "Measured Value": f"{mean_idle_power_w:.2f}" if mean_idle_power_w else "N/A", "Unit": "Watts (W)"},
        {"Metric Parameter": "Unit Energy Cost per Token", "Measured Value": f"{mj_per_tok:.3f}", "Unit": "mJ / token"},
        {"Metric Parameter": "Unit Energy Cost per Query", "Measured Value": f"{j_per_query:.3f}", "Unit": "Joules / query"},
        {"Metric Parameter": "Overall Macro Accuracy Across All Tasks", "Measured Value": f"{overall_acc:.2f}%", "Unit": "percent"}
    ]

    t2_df = pd.DataFrame(data)

    csv_path = os.path.join(output_dir, "table2_model_execution_energy.csv")
    t2_df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = os.path.join(output_dir, "table2_model_execution_energy.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Table 2: Execution Time & Energy Summary for Model: {model_name}\n\n")
        f.write(t2_df.to_markdown(index=False))
        f.write("\n")

    tex_path = os.path.join(output_dir, "table2_model_execution_energy.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 2: Model Execution & Energy Summary\n")
        f.write(t2_df.to_latex(index=False, escape=True))
        f.write("\n")

    return t2_df


# ============================================================
# Seaborn Visualization Generator
# ============================================================

def generate_seaborn_figures(df: pd.DataFrame, records: List[Dict[str, Any]], plots_dir: str, model_name: str, hw_name: str) -> None:
    """Generates publication-quality Seaborn figures at 300 DPI."""
    if df.empty or not records:
        return

    os.makedirs(plots_dir, exist_ok=True)
    raw_df = pd.DataFrame(records)

    # Set Seaborn global theme
    sns.set_theme(style="whitegrid", font_scale=1.1, palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8
    })

    # Filter primary experiment strategies for strategy plots
    prompt_strats = ["zero_shot_direct", "few_shot_3", "zero_shot_cot", "short_cot", "long_cot"]
    prompt_df = df[df["Raw Strategy"].isin(prompt_strats)].copy()

    # -------------------------------------------------------------
    # Figure 1: Model Accuracy by Dataset and Prompt Strategy
    # -------------------------------------------------------------
    if not prompt_df.empty and prompt_df["Accuracy (%)"].notna().any():
        plt.figure(figsize=(10, 5.5), dpi=300)
        ax = sns.barplot(
            data=prompt_df,
            x="Dataset",
            y="Accuracy (%)",
            hue="Strategy",
            palette="Set2",
            edgecolor="#222222",
            linewidth=0.7
        )
        plt.title(f"Task Accuracy by Prompting Strategy — Model: {model_name} ({hw_name})", fontsize=13, weight="bold", pad=12)
        plt.xlabel("Benchmark Task Dataset", fontsize=11, labelpad=8)
        plt.ylabel("Accuracy (%)", fontsize=11, labelpad=8)
        plt.ylim(0, max(100.0, (prompt_df["Accuracy (%)"].max() or 0) + 10))
        plt.legend(title="Prompting Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "1_model_accuracy_by_dataset_and_strategy.png"), dpi=300)
        plt.savefig(os.path.join(plots_dir, "1_model_accuracy_by_dataset_and_strategy.pdf"))
        plt.close()

    # -------------------------------------------------------------
    # Figure 2: Latency & TTFT Scaling by Strategy and Context
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 2a: Strategy Latencies
    if not prompt_df.empty and prompt_df["Total Latency (ms)"].notna().any():
        sns.barplot(
            data=prompt_df,
            x="Strategy",
            y="Total Latency (ms)",
            hue="Dataset",
            ax=ax1,
            palette="muted",
            edgecolor="#222222",
            linewidth=0.6
        )
        ax1.set_title("Total Latency by Prompting Strategy", fontsize=12, weight="bold")
        ax1.set_xlabel("Strategy", fontsize=10)
        ax1.set_ylabel("Total Latency (ms)", fontsize=10)
        ax1.tick_params(axis='x', rotation=30)
        ax1.legend(title="Dataset", fontsize=9)

    # 2b: Context Length Scaling Latency
    ctx_df = df[df["Raw Strategy"].str.startswith("ctx_")].copy()
    if not ctx_df.empty and ctx_df["TTFT (ms)"].notna().any():
        # Parse context token integer
        ctx_df["Context_Tokens"] = ctx_df["Raw Strategy"].str.replace("ctx_", "").astype(int)
        ctx_df = ctx_df.sort_values("Context_Tokens")
        
        sns.lineplot(
            data=ctx_df,
            x="Context_Tokens",
            y="TTFT (ms)",
            hue="Dataset",
            marker="o",
            markersize=8,
            linewidth=2.2,
            ax=ax2,
            palette="tab10"
        )
        ax2.set_title("Time-to-First-Token (TTFT) vs. Context Length", fontsize=12, weight="bold")
        ax2.set_xlabel("Context Length (Tokens)", fontsize=10)
        ax2.set_ylabel("TTFT Prefill Latency (ms)", fontsize=10)
        ax2.legend(title="Dataset", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Context Scaling Data Not Available", ha="center", va="center", transform=ax2.transAxes)

    plt.suptitle(f"Latency & Prefill Scaling Profile — {model_name}", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_latency_by_strategy_and_context.png"), dpi=300)
    plt.savefig(os.path.join(plots_dir, "2_latency_by_strategy_and_context.pdf"))
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Energy vs. Accuracy Pareto Trade-off
    # -------------------------------------------------------------
    valid_pareto = df[df["Total Energy (J)"].notna() & df["Accuracy (%)"].notna()].copy()
    if not valid_pareto.empty:
        plt.figure(figsize=(9, 6), dpi=300)
        
        sns.scatterplot(
            data=valid_pareto,
            x="Total Energy (J)",
            y="Accuracy (%)",
            hue="Dataset",
            style="Experiment Type",
            s=120,
            palette="bright",
            edgecolor="#111111",
            alpha=0.9
        )

        # Plot Pareto frontier
        points = [{"energy": float(r["Total Energy (J)"]), "accuracy": float(r["Accuracy (%)"]), "strategy": r["Strategy"]} for _, r in valid_pareto.iterrows()]
        frontier = compute_pareto_frontier(points, maximize_keys=["accuracy"], minimize_keys=["energy"])
        if frontier:
            frontier = sorted(frontier, key=lambda p: p["energy"])
            f_energies = [p["energy"] for p in frontier]
            f_accuracies = [p["accuracy"] for p in frontier]
            plt.plot(f_energies, f_accuracies, color="#E11D48", linestyle="--", linewidth=2.0, label="Pareto Frontier", zorder=3)

        plt.title(f"Energy vs. Accuracy Trade-Off & Pareto Frontier — {model_name}", fontsize=13, weight="bold", pad=12)
        plt.xlabel("Mean Energy Consumption per Query (Joules)", fontsize=11, labelpad=8)
        plt.ylabel("Accuracy (%)", fontsize=11, labelpad=8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "3_energy_vs_accuracy_pareto_tradeoff.png"), dpi=300)
        plt.savefig(os.path.join(plots_dir, "3_energy_vs_accuracy_pareto_tradeoff.pdf"))
        plt.close()

    # -------------------------------------------------------------
    # Figure 4: Prefill vs. Decode Energy Breakdown
    # -------------------------------------------------------------
    if not prompt_df.empty and prompt_df["Prefill Energy (J)"].notna().any() and prompt_df["Decode Energy (J)"].notna().any():
        plt.figure(figsize=(10, 5.5), dpi=300)
        
        # Melt for Seaborn stacked/grouped visualization
        melted = prompt_df.melt(
            id_vars=["Strategy", "Dataset"],
            value_vars=["Prefill Energy (J)", "Decode Energy (J)"],
            var_name="Phase",
            value_name="Energy (J)"
        )
        
        sns.barplot(
            data=melted,
            x="Strategy",
            y="Energy (J)",
            hue="Phase",
            palette={"Prefill Energy (J)": "#0284C7", "Decode Energy (J)": "#F97316"},
            edgecolor="#222222",
            linewidth=0.7
        )
        plt.title(f"Phase Energy Decomposition (Prefill vs. Decode) — {model_name}", fontsize=13, weight="bold", pad=12)
        plt.xlabel("Prompting Strategy", fontsize=11, labelpad=8)
        plt.ylabel("Energy (Joules)", fontsize=11, labelpad=8)
        plt.xticks(rotation=25)
        plt.legend(title="Inference Phase")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "4_energy_breakdown_prefill_decode.png"), dpi=300)
        plt.savefig(os.path.join(plots_dir, "4_energy_breakdown_prefill_decode.pdf"))
        plt.close()

    # -------------------------------------------------------------
    # Figure 5: RAG Energy and Accuracy Scaling
    # -------------------------------------------------------------
    rag_df = df[df["Raw Strategy"].str.startswith("rag_") | (df["Raw Strategy"] == "zero_shot_direct")].copy()
    if not rag_df.empty and len(rag_df) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        sns.barplot(
            data=rag_df,
            x="Strategy",
            y="Accuracy (%)",
            hue="Dataset",
            ax=ax1,
            palette="coolwarm",
            edgecolor="#222222",
            linewidth=0.7
        )
        ax1.set_title("RAG Accuracy Scaling (Top-k)", fontsize=12, weight="bold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.tick_params(axis='x', rotation=25)

        sns.barplot(
            data=rag_df,
            x="Strategy",
            y="Total Energy (J)",
            hue="Dataset",
            ax=ax2,
            palette="viridis",
            edgecolor="#222222",
            linewidth=0.7
        )
        ax2.set_title("RAG Energy Scaling (Top-k)", fontsize=12, weight="bold")
        ax2.set_ylabel("Energy (Joules)")
        ax2.tick_params(axis='x', rotation=25)

        plt.suptitle(f"BM25 Retrieval-Augmented Generation Scaling — {model_name}", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "5_rag_energy_and_accuracy_scaling.png"), dpi=300)
        plt.savefig(os.path.join(plots_dir, "5_rag_energy_and_accuracy_scaling.pdf"))
        plt.close()

    # -------------------------------------------------------------
    # Figure 6: Token Generation Throughput & Runtime
    # -------------------------------------------------------------
    if "total_latency_ms" in raw_df.columns and "output_tokens" in raw_df.columns:
        valid_raw = raw_df[raw_df["total_latency_ms"].notna() & raw_df["output_tokens"].notna() & (raw_df["total_latency_ms"] > 0)].copy()
        if not valid_raw.empty:
            valid_raw["throughput_tok_s"] = valid_raw["output_tokens"] / (valid_raw["total_latency_ms"] / 1000.0)
            
            plt.figure(figsize=(10, 5), dpi=300)
            ds_col = "dataset" if "dataset" in valid_raw.columns else "task_type"
            sns.boxplot(
                data=valid_raw,
                x=ds_col,
                y="throughput_tok_s",
                hue=ds_col,
                legend=False,
                palette="Accent",
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
            )
            plt.title(f"Token Generation Throughput Distribution — {model_name}", fontsize=13, weight="bold", pad=12)
            plt.xlabel("Benchmark Dataset", fontsize=11, labelpad=8)
            plt.ylabel("Throughput (Tokens / Second)", fontsize=11, labelpad=8)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "6_model_runtime_token_throughput.png"), dpi=300)
            plt.savefig(os.path.join(plots_dir, "6_model_runtime_token_throughput.pdf"))
            plt.close()

    # -------------------------------------------------------------
    # Figure 7: Single-LLM Comprehensive Executive Dashboard
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # Panel A: Accuracy Heatmap / Barplot
    if not prompt_df.empty:
        try:
            pivot_acc = prompt_df.pivot(index="Strategy", columns="Dataset", values="Accuracy (%)")
            sns.heatmap(pivot_acc, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0, 0], cbar_kws={'label': 'Accuracy (%)'})
            axes[0, 0].set_title("(A) Accuracy (%) by Prompting Strategy", fontsize=12, weight="bold")
        except Exception:
            sns.barplot(data=prompt_df, x="Strategy", y="Accuracy (%)", hue="Strategy", legend=False, ax=axes[0, 0], palette="Set2")
            axes[0, 0].set_title("(A) Accuracy (%)", fontsize=12, weight="bold")
            axes[0, 0].tick_params(axis='x', rotation=25)

    # Panel B: Energy Consumption
    if not prompt_df.empty:
        try:
            pivot_eng = prompt_df.pivot(index="Strategy", columns="Dataset", values="Total Energy (J)")
            sns.heatmap(pivot_eng, annot=True, fmt=".2f", cmap="OrRd", ax=axes[0, 1], cbar_kws={'label': 'Energy (J)'})
            axes[0, 1].set_title("(B) Energy Consumption (Joules)", fontsize=12, weight="bold")
        except Exception:
            sns.barplot(data=prompt_df, x="Strategy", y="Total Energy (J)", hue="Strategy", legend=False, ax=axes[0, 1], palette="Reds")
            axes[0, 1].set_title("(B) Energy (J)", fontsize=12, weight="bold")
            axes[0, 1].tick_params(axis='x', rotation=25)

    # Panel C: Energy vs Accuracy Pareto
    if not valid_pareto.empty:
        sns.scatterplot(
            data=valid_pareto,
            x="Total Energy (J)",
            y="Accuracy (%)",
            hue="Dataset",
            s=100,
            ax=axes[1, 0],
            palette="tab10"
        )
        axes[1, 0].set_title("(C) Energy-Accuracy Frontier", fontsize=12, weight="bold")

    # Panel D: Latency Profile
    if not prompt_df.empty:
        sns.barplot(
            data=prompt_df,
            x="Strategy",
            y="Total Latency (ms)",
            hue="Strategy",
            legend=False,
            ax=axes[1, 1],
            palette="Blues_d"
        )
        axes[1, 1].set_title("(D) Mean Total Latency (ms)", fontsize=12, weight="bold")
        axes[1, 1].tick_params(axis='x', rotation=25)

    plt.suptitle(f"PromptEnergy-Bench Executive Model Dashboard: {model_name} on {hw_name}", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "7_comprehensive_model_dashboard.png"), dpi=300)
    plt.savefig(os.path.join(plots_dir, "7_comprehensive_model_dashboard.pdf"))
    plt.close()


# ============================================================
# Main Single-Model Report Orchestrator
# ============================================================

def generate_single_model_report(model_run_dir: str, model_name: str, hw_name: str) -> Dict[str, Any]:
    """Orchestrates ingestion, Table 1 & Table 2 generation, and Seaborn plots for a single completed LLM."""
    print(f"\n{'='*75}")
    print(f"GENERATING AGGREGATED REPORT FOR MODEL: {model_name}")
    print(f"Target Directory: {model_run_dir}")
    print(f"{'='*75}")

    records, meta = collect_model_records(model_run_dir)
    if not records:
        print(f"[WARNING] No records found in {model_run_dir}. Skipping report generation.")
        return {}

    tables_dir = os.path.join(model_run_dir, "tables")
    plots_dir = os.path.join(model_run_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Generate Table 1: Full Merged Results Table
    df1 = generate_table1_full_merged(records, tables_dir, model_name)
    print(f"  [OK] Table 1 (Full Merged Results) -> {os.path.join(tables_dir, 'table1_full_merged_results.csv')}")

    # 2. Generate Table 2: Execution Time & Energy Summary Table
    df2 = generate_table2_execution_energy(records, meta, tables_dir, model_name, hw_name)
    print(f"  [OK] Table 2 (Model Execution & Energy) -> {os.path.join(tables_dir, 'table2_model_execution_energy.csv')}")

    # 3. Generate Seaborn Figures
    generate_seaborn_figures(df1, records, plots_dir, model_name, hw_name)
    print(f"  [OK] Seaborn Figures (PNG @ 300 DPI + PDF) -> {plots_dir}")

    # 4. Save summary_all_experiments.json in model root
    summary_json_path = os.path.join(model_run_dir, "summary_all_experiments.json")
    model_summary = {
        "model_name": model_name,
        "clean_tag": get_clean_model_tag(model_name),
        "hardware": hw_name,
        "total_records": len(records),
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tables": {
            "table1_csv": os.path.join(tables_dir, "table1_full_merged_results.csv"),
            "table2_csv": os.path.join(tables_dir, "table2_model_execution_energy.csv")
        },
        "plots_directory": plots_dir
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=4)

    print(f"  [OK] Consolidated Model Summary -> {summary_json_path}")
    print(f"[SUCCESS] Completed all report artifacts for {model_name}!\n")

    return model_summary
