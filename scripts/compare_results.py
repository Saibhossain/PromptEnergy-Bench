#!/usr/bin/env python3
"""Interactive & CLI Result Comparison Suite for PromptEnergy-Bench.

Compares benchmark results across models (e.g., qwen3.5:0.8b-mlx vs qwen3.5:2b-mlx),
prompting strategies, datasets, context lengths, and hardware environments.

Generates:
1. Publication-Grade Comparative Tables (CSV, Markdown, LaTeX)
2. High-Resolution Visualizations (PNG @ 300 DPI, PDF Vector, SVG)

Usage:
  # Interactive mode (User selects files or folders interactively):
  python scripts/compare_results.py

  # CLI mode (Pass specific run directories or results.jsonl files):
  python scripts/compare_results.py --inputs results/primary_exp_gsm8k/macbook_air_m1/* results/context_scaling_gsm8k/macbook_air_m1/*

  # Compare everything under results/:
  python scripts/compare_results.py --inputs results/ --output-dir results/comparison_report
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.pareto import compute_pareto_frontier
from src.analysis.statistics import calculate_summary_statistics

# Publication Styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.autolayout": True
})

PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Green
    "#CC79A7",  # Purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#333333"   # Charcoal
]


# ============================================================
# Interactive Run & Folder Discovery
# ============================================================

def discover_available_runs(root_dir: str = "results") -> List[Dict[str, Any]]:
    """Scans root_dir for all experiment runs containing results.jsonl."""
    discovered = []
    if not os.path.exists(root_dir):
        return discovered

    for root, dirs, files in os.walk(root_dir):
        if "results.jsonl" in files or "raw_results.jsonl" in files:
            target_file = "results.jsonl" if "results.jsonl" in files else "raw_results.jsonl"
            full_path = os.path.join(root, target_file)
            size_kb = os.path.getsize(full_path) / 1024
            
            # Read first line to get metadata
            model_name = "unknown"
            exp_name = "unknown"
            device = "unknown"
            dataset = "unknown"
            count = 0
            try:
                with open(full_path, "r", encoding="utf-8") as h:
                    for idx, line in enumerate(h):
                        if idx == 0:
                            data = json.loads(line)
                            model_name = data.get("model") or model_name
                            exp_name = data.get("experiment_name") or exp_name
                            device = data.get("device") or data.get("hardware_identifier") or device
                            dataset = data.get("dataset") or data.get("task_type") or dataset
                        count += 1
            except Exception:
                pass

            discovered.append({
                "dir": root,
                "file": full_path,
                "model": model_name,
                "experiment": exp_name,
                "device": device,
                "dataset": dataset,
                "records": count,
                "size_kb": round(size_kb, 1)
            })

    return sorted(discovered, key=lambda x: x["dir"])


def interactive_selection(discovered: List[Dict[str, Any]]) -> List[str]:
    """Prompts the user interactively in the terminal to choose which runs to compare."""
    if not discovered:
        print("[WARN] No completed experiment runs found in results/. Please run benchmarks first.")
        return []

    print("\n" + "=" * 80)
    print("PROMPTENERGY-BENCH: INTERACTIVE COMPARISON SELECTION")
    print("=" * 80)
    print(f"{'#':<3} | {'Experiment':<22} | {'Model':<18} | {'Device':<16} | {'Samples':<7} | Path")
    print("-" * 80)

    for i, run in enumerate(discovered, 1):
        rel_path = os.path.relpath(run["dir"])
        print(f"{i:<3} | {run['experiment']:<22} | {run['model']:<18} | {run['device']:<16} | {run['records']:<7} | {rel_path}")

    print("-" * 80)
    print("Enter run numbers separated by spaces (e.g. '1 2 3'), 'all' to select all, or 'q' to quit.")
    print("=" * 80)

    while True:
        try:
            choice = input("Select runs to compare [all]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return []

        if not choice or choice.lower() == "all":
            return [r["file"] for r in discovered]
        if choice.lower() in ("q", "quit", "exit"):
            return []

        selected_indices = []
        valid = True
        for part in choice.replace(",", " ").split():
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(discovered):
                    selected_indices.append(idx - 1)
                else:
                    print(f"Error: Index {idx} out of range (1 - {len(discovered)}).")
                    valid = False
            else:
                print(f"Error: Invalid input '{part}'.")
                valid = False

        if valid and selected_indices:
            return [discovered[i]["file"] for i in selected_indices]


# ============================================================
# Record Loading & Aggregation
# ============================================================

def load_records(inputs: List[str]) -> List[Dict[str, Any]]:
    """Loads all records from specified file paths or directory trees."""
    all_records = []
    seen_keys: Set[str] = set()

    expanded_paths = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f in ("results.jsonl", "raw_results.jsonl"):
                        expanded_paths.append(os.path.join(root, f))
        elif os.path.isfile(p):
            expanded_paths.append(p)
        else:
            # Handle glob patterns
            matched = glob.glob(p, recursive=True)
            for m in matched:
                if os.path.isdir(m):
                    for root, _, files in os.walk(m):
                        for f in files:
                            if f in ("results.jsonl", "raw_results.jsonl"):
                                expanded_paths.append(os.path.join(root, f))
                elif os.path.isfile(m):
                    expanded_paths.append(m)

    for fpath in expanded_paths:
        try:
            with open(fpath, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ckey = rec.get("condition_key") or f"{rec.get('run_id')}_{rec.get('sample_id')}_{rec.get('strategy')}_{rec.get('repetition', 1)}"
                        if ckey not in seen_keys:
                            seen_keys.add(ckey)
                            all_records.append(rec)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Warning: Failed reading {fpath}: {e}")

    return all_records


def parse_numeric(val: Any) -> Optional[float]:
    if val is None or val == "N/A" or val == "unavailable" or val == "":
        return None
    try:
        f = float(val)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


# ============================================================
# Table Generators (CSV, Markdown, LaTeX)
# ============================================================

def export_table(rows: List[Dict[str, Any]], fieldnames: List[str], base_path: str, title: str = "") -> None:
    """Exports a table into CSV, Markdown, and LaTeX formats."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # 1. CSV
    csv_file = f"{base_path}.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "N/A") for k in fieldnames})

    # 2. Markdown
    md_file = f"{base_path}.md"
    with open(md_file, "w", encoding="utf-8") as mf:
        if title:
            mf.write(f"# {title}\n\n")
        mf.write("| " + " | ".join(fieldnames) + " |\n")
        mf.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for r in rows:
            mf.write("| " + " | ".join(str(r.get(k, "N/A")) for k in fieldnames) + " |\n")

    # 3. LaTeX
    tex_file = f"{base_path}.tex"
    with open(tex_file, "w", encoding="utf-8") as tf:
        tf.write("% " + title + "\n")
        col_align = "l" + "r" * (len(fieldnames) - 1)
        tf.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        tf.write(f"\\begin{{tabular}}{{{col_align}}}\n\\toprule\n")
        tf.write(" & ".join(fieldnames).replace("_", "\\_") + " \\\\\n\\midrule\n")
        for r in rows:
            vals = [str(r.get(k, "N/A")).replace("_", "\\_").replace("%", "\\%") for k in fieldnames]
            tf.write(" & ".join(vals) + " \\\\\n")
        tf.write("\\bottomrule\n\\end{tabular}\n")
        tf.write(f"\\caption{{{title}}}\n\\label{{tab:{os.path.basename(base_path)}}}\n\\end{{table*}}\n")


def generate_comparative_tables(records: List[Dict[str, Any]], output_dir: str) -> None:
    """Generates all comparative tables across models, strategies, datasets, and hardware."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Pre-calculate Prefill and Decode Energy per record if not present
    for r in records:
        e_tot = parse_numeric(r.get("energy_metrics", {}).get("energy_total_j") if isinstance(r.get("energy_metrics"), dict) else r.get("energy_total_j"))
        ttft = parse_numeric(r.get("latency_metrics", {}).get("ttft_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("ttft_ms"))
        tot_lat = parse_numeric(r.get("latency_metrics", {}).get("total_latency_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("total_latency_ms"))
        gen_lat = parse_numeric(r.get("latency_metrics", {}).get("generation_latency_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("generation_latency_ms"))

        if e_tot is not None and tot_lat is not None and tot_lat > 0:
            if "energy_prefill_j" not in r or r["energy_prefill_j"] is None:
                r["energy_prefill_j"] = e_tot * ((ttft or 0.0) / tot_lat)
            if "energy_decode_j" not in r or r["energy_decode_j"] is None:
                gen_time = gen_lat if gen_lat is not None else (tot_lat - (ttft or 0.0))
                r["energy_decode_j"] = e_tot * (gen_time / tot_lat)

    # 1. Model Comparison Table
    model_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        m = r.get("model", "unknown")
        model_groups.setdefault(m, []).append(r)

    model_rows = []
    for m, m_recs in model_groups.items():
        devs = sorted(list(set(r.get("device") or r.get("hardware_identifier") or "unknown" for r in m_recs)))
        e_vals = [parse_numeric(r.get("energy_total_j")) for r in m_recs if parse_numeric(r.get("energy_total_j")) is not None]
        pref_e_vals = [parse_numeric(r.get("energy_prefill_j")) for r in m_recs if parse_numeric(r.get("energy_prefill_j")) is not None]
        dec_e_vals = [parse_numeric(r.get("energy_decode_j")) for r in m_recs if parse_numeric(r.get("energy_decode_j")) is not None]

        t_vals = [parse_numeric(r.get("total_latency_ms") or (r.get("latency_metrics", {}).get("total_latency_ms") if isinstance(r.get("latency_metrics"), dict) else None)) for r in m_recs]
        t_vals = [t for t in t_vals if t is not None]

        ttft_vals = [parse_numeric(r.get("ttft_ms") or (r.get("latency_metrics", {}).get("ttft_ms") if isinstance(r.get("latency_metrics"), dict) else None)) for r in m_recs]
        ttft_vals = [tt for tt in ttft_vals if tt is not None]

        in_toks = [parse_numeric(r.get("input_tokens") or r.get("input_token_count")) for r in m_recs]
        in_toks = [it for it in in_toks if it is not None]

        out_toks = [parse_numeric(r.get("output_tokens") or r.get("output_token_count")) for r in m_recs]
        out_toks = [ot for ot in out_toks if ot is not None]

        acc_vals = [1 if r.get("answer_correct") is True else 0 for r in m_recs if r.get("answer_correct") is not None]

        # Unit token energy (mJ / token)
        sum_pref_e = sum(pref_e_vals) if pref_e_vals else 0.0
        sum_dec_e = sum(dec_e_vals) if dec_e_vals else 0.0
        sum_in_t = sum(in_toks) if in_toks else 0
        sum_out_t = sum(out_toks) if out_toks else 0

        pref_mj = f"{(sum_pref_e / sum_in_t) * 1000.0:.2f}" if (sum_pref_e > 0 and sum_in_t > 0) else "N/A"
        dec_mj = f"{(sum_dec_e / sum_out_t) * 1000.0:.2f}" if (sum_dec_e > 0 and sum_out_t > 0) else "N/A"

        # Decode Throughput (tok/s)
        tps_list = []
        for r in m_recs:
            ot = parse_numeric(r.get("output_tokens") or r.get("output_token_count"))
            lat = parse_numeric(r.get("generation_latency_ms") or r.get("total_latency_ms"))
            if ot is not None and lat is not None and lat > 0:
                tps_list.append(ot / (lat / 1000.0))
        mean_tps = f"{np.mean(tps_list):.2f}" if tps_list else "N/A"

        model_rows.append({
            "Model": m,
            "Hardware": ", ".join(devs),
            "Samples": len(m_recs),
            "Accuracy (%)": f"{np.mean(acc_vals)*100:.2f}%" if acc_vals else "N/A",
            "Total Energy (J)": f"{np.mean(e_vals):.3f}" if e_vals else "N/A",
            "Prefill Energy (J)": f"{np.mean(pref_e_vals):.3f}" if pref_e_vals else "N/A",
            "Decode Energy (J)": f"{np.mean(dec_e_vals):.3f}" if dec_e_vals else "N/A",
            "Prefill (mJ/tok)": pref_mj,
            "Decode (mJ/tok)": dec_mj,
            "Decode TPS (tok/s)": mean_tps,
            "Mean TTFT (ms)": f"{np.mean(ttft_vals):.1f}" if ttft_vals else "N/A",
            "Mean Latency (ms)": f"{np.mean(t_vals):.1f}" if t_vals else "N/A"
        })

    export_table(
        model_rows,
        ["Model", "Hardware", "Samples", "Accuracy (%)", "Total Energy (J)", "Prefill Energy (J)", "Decode Energy (J)", "Prefill (mJ/tok)", "Decode (mJ/tok)", "Decode TPS (tok/s)", "Mean TTFT (ms)", "Mean Latency (ms)"],
        os.path.join(tables_dir, "1_model_comparison"),
        "Cross-Model Comparative Performance and Energy Decomposition Summary"
    )

    # 2. Strategy Comparison Table
    strat_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        m = r.get("model", "unknown")
        s = r.get("strategy", "unknown")
        strat_groups.setdefault((m, s), []).append(r)

    strat_rows = []
    for (m, s), s_recs in sorted(strat_groups.items()):
        e_vals = [parse_numeric(r.get("energy_total_j")) for r in s_recs if parse_numeric(r.get("energy_total_j")) is not None]
        pref_e_vals = [parse_numeric(r.get("energy_prefill_j")) for r in s_recs if parse_numeric(r.get("energy_prefill_j")) is not None]
        dec_e_vals = [parse_numeric(r.get("energy_decode_j")) for r in s_recs if parse_numeric(r.get("energy_decode_j")) is not None]

        t_vals = [parse_numeric(r.get("total_latency_ms")) for r in s_recs if parse_numeric(r.get("total_latency_ms")) is not None]
        ttft_vals = [parse_numeric(r.get("ttft_ms")) for r in s_recs if parse_numeric(r.get("ttft_ms")) is not None]

        in_toks = [parse_numeric(r.get("input_tokens") or r.get("input_token_count")) for r in s_recs]
        in_toks = [it for it in in_toks if it is not None]

        tok_vals = [parse_numeric(r.get("output_tokens") or r.get("output_token_count")) for r in s_recs]
        tok_vals = [tk for tk in tok_vals if tk is not None]

        acc_vals = [1 if r.get("answer_correct") is True else 0 for r in s_recs if r.get("answer_correct") is not None]

        tot_e_mean = np.mean(e_vals) if e_vals else 0.0
        pref_e_mean = np.mean(pref_e_vals) if pref_e_vals else 0.0
        dec_e_mean = np.mean(dec_e_vals) if dec_e_vals else 0.0

        pref_share = f"{(pref_e_mean / tot_e_mean) * 100.0:.1f}%" if tot_e_mean > 0 else "N/A"
        dec_share = f"{(dec_e_mean / tot_e_mean) * 100.0:.1f}%" if tot_e_mean > 0 else "N/A"

        sum_pref_e = sum(pref_e_vals) if pref_e_vals else 0.0
        sum_dec_e = sum(dec_e_vals) if dec_e_vals else 0.0
        sum_in_t = sum(in_toks) if in_toks else 0
        sum_out_t = sum(tok_vals) if tok_vals else 0

        pref_mj = f"{(sum_pref_e / sum_in_t) * 100.0:.2f}" if (sum_pref_e > 0 and sum_in_t > 0) else "N/A"
        dec_mj = f"{(sum_dec_e / sum_out_t) * 100.0:.2f}" if (sum_dec_e > 0 and sum_out_t > 0) else "N/A"

        strat_rows.append({
            "Model": m,
            "Strategy": s,
            "Samples": len(s_recs),
            "Accuracy (%)": f"{np.mean(acc_vals)*100:.2f}%" if acc_vals else "N/A",
            "Total Energy (J)": f"{tot_e_mean:.3f}",
            "Prefill Energy (J)": f"{pref_e_mean:.3f}",
            "Decode Energy (J)": f"{dec_e_mean:.3f}",
            "Prefill Share": pref_share,
            "Decode Share": dec_share,
            "Prefill (mJ/tok)": pref_mj,
            "Decode (mJ/tok)": dec_mj,
            "Mean Input Tokens": f"{np.mean(in_toks):.1f}" if in_toks else "N/A",
            "Mean Output Tokens": f"{np.mean(tok_vals):.1f}" if tok_vals else "N/A",
            "Mean TTFT (ms)": f"{np.mean(ttft_vals):.1f}" if ttft_vals else "N/A",
            "Mean Latency (ms)": f"{np.mean(t_vals):.1f}" if t_vals else "N/A"
        })

    export_table(
        strat_rows,
        ["Model", "Strategy", "Samples", "Accuracy (%)", "Total Energy (J)", "Prefill Energy (J)", "Decode Energy (J)", "Prefill Share", "Decode Share", "Prefill (mJ/tok)", "Decode (mJ/tok)", "Mean Input Tokens", "Mean Output Tokens", "Mean TTFT (ms)", "Mean Latency (ms)"],
        os.path.join(tables_dir, "2_strategy_comparison"),
        "Prompting Strategy Performance & Energy Decomposition Breakdown"
    )

    # 3. Marginal Energy Gain (MEG) Table
    meg_rows = []
    for m in model_groups.keys():
        m_strats = {s: recs for (mod, s), recs in strat_groups.items() if mod == m}
        baseline_key = "zero_shot_direct" if "zero_shot_direct" in m_strats else None
        if not baseline_key and m_strats:
            baseline_key = list(m_strats.keys())[0]

        if baseline_key:
            b_recs = m_strats[baseline_key]
            b_acc = np.mean([1 if r.get("answer_correct") is True else 0 for r in b_recs]) if b_recs else 0.0
            b_e_list = [parse_numeric(r.get("energy_total_j")) for r in b_recs if parse_numeric(r.get("energy_total_j")) is not None]
            b_e = np.mean(b_e_list) if b_e_list else 0.0

            for s, s_recs in m_strats.items():
                if s == baseline_key:
                    continue
                s_acc = np.mean([1 if r.get("answer_correct") is True else 0 for r in s_recs]) if s_recs else 0.0
                s_e_list = [parse_numeric(r.get("energy_total_j")) for r in s_recs if parse_numeric(r.get("energy_total_j")) is not None]
                s_e = np.mean(s_e_list) if s_e_list else 0.0

                delta_acc = (s_acc - b_acc) * 100.0
                delta_e = s_e - b_e
                meg_val = (delta_acc / delta_e) if delta_e > 0.0001 else 0.0

                meg_rows.append({
                    "Model": m,
                    "Baseline": baseline_key,
                    "Strategy": s,
                    "Delta Acc (% pts)": f"{delta_acc:+.2f}%",
                    "Delta Energy (J)": f"{delta_e:+.3f} J",
                    "MEG (% / J)": f"{meg_val:.3f} %/J"
                })

    if meg_rows:
        export_table(
            meg_rows,
            ["Model", "Baseline", "Strategy", "Delta Acc (% pts)", "Delta Energy (J)", "MEG (% / J)"],
            os.path.join(tables_dir, "3_marginal_energy_gain"),
            "Marginal Energy Gain (MEG) across Prompting Strategies"
        )

    print(f"  [TABLES] Exported comparative tables to: {tables_dir}/")


# ============================================================
# High-Resolution Publication Figures (PNG, PDF, SVG)
# ============================================================

def save_plot(fig: plt.Figure, plots_dir: str, base_name: str) -> None:
    """Saves high-res publication figures in PNG (300 DPI), PDF, and SVG formats."""
    os.makedirs(plots_dir, exist_ok=True)
    png_path = os.path.join(plots_dir, f"{base_name}.png")
    pdf_path = os.path.join(plots_dir, f"{base_name}.pdf")
    svg_path = os.path.join(plots_dir, f"{base_name}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIGURE] Saved {base_name} (PNG, PDF, SVG)")


def generate_comparative_figures(records: List[Dict[str, Any]], output_dir: str) -> None:
    """Generates all comparative figures across models, strategies, context lengths, and RAG."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Pre-calculate Prefill and Decode Energy per record if not present
    for r in records:
        e_tot = parse_numeric(r.get("energy_metrics", {}).get("energy_total_j") if isinstance(r.get("energy_metrics"), dict) else r.get("energy_total_j"))
        ttft = parse_numeric(r.get("latency_metrics", {}).get("ttft_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("ttft_ms"))
        tot_lat = parse_numeric(r.get("latency_metrics", {}).get("total_latency_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("total_latency_ms"))
        gen_lat = parse_numeric(r.get("latency_metrics", {}).get("generation_latency_ms") if isinstance(r.get("latency_metrics"), dict) else r.get("generation_latency_ms"))

        if e_tot is not None and tot_lat is not None and tot_lat > 0:
            if "energy_prefill_j" not in r or r["energy_prefill_j"] is None:
                r["energy_prefill_j"] = e_tot * ((ttft or 0.0) / tot_lat)
            if "energy_decode_j" not in r or r["energy_decode_j"] is None:
                gen_time = gen_lat if gen_lat is not None else (tot_lat - (ttft or 0.0))
                r["energy_decode_j"] = e_tot * (gen_time / tot_lat)

    models = sorted(list(set(r.get("model", "unknown") for r in records)))
    strategies = sorted(list(set(r.get("strategy", "unknown") for r in records)))

    # 1. Figure: Energy vs Accuracy Pareto Curves by Model
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    offset_cycle = [(10, 8), (-12, 14), (12, -14), (-14, -14), (0, 18), (0, -20)]

    for idx, m in enumerate(models):
        m_recs = [r for r in records if r.get("model") == m]
        strat_points = []
        for s in strategies:
            s_recs = [r for r in m_recs if r.get("strategy") == s]
            if not s_recs:
                continue
            e_vals = [parse_numeric(r.get("energy_total_j")) for r in s_recs if parse_numeric(r.get("energy_total_j")) is not None]
            acc_vals = [1 if r.get("answer_correct") is True else 0 for r in s_recs if r.get("answer_correct") is not None]
            if e_vals and acc_vals:
                strat_points.append((np.mean(e_vals), np.mean(acc_vals) * 100.0, s))

        if strat_points:
            color = PALETTE[idx % len(PALETTE)]
            xs = [p[0] for p in strat_points]
            ys = [p[1] for p in strat_points]
            labels = [p[2] for p in strat_points]
            ax.scatter(xs, ys, label=m, color=color, s=120, alpha=0.9, edgecolors="black", linewidths=1.0, zorder=5)

            for p_idx, (x, y, lab) in enumerate(zip(xs, ys, labels)):
                off = offset_cycle[p_idx % len(offset_cycle)]
                if y < 1.0:
                    off = (off[0], 10 + (p_idx % 3) * 14)
                ax.annotate(
                    lab.replace("_", " "),
                    (x, y),
                    textcoords="offset points",
                    xytext=off,
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="#CCCCCC", linewidth=0.6),
                    arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.6, alpha=0.6) if abs(off[0]) > 14 or abs(off[1]) > 14 else None,
                    zorder=6
                )

            sorted_pts = sorted(strat_points, key=lambda p: p[0])
            ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts], color=color, linestyle="--", linewidth=1.8, alpha=0.7)

    ax.set_xlabel("Mean Total Energy per Query (Joules)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Energy-Accuracy Trade-off & Pareto Frontier by Model", fontsize=12, fontweight="semibold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    save_plot(fig, plots_dir, "01_energy_vs_accuracy_pareto")

    # 2. Figure: Prefill vs Decode Stacked Energy Comparison
    if len(strategies) > 0:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        strat_disp = [s.replace("_", "\n") for s in strategies]
        x = np.arange(len(strategies))
        width = 0.52

        pref_means = []
        dec_means = []
        tot_means = []

        for s in strategies:
            s_recs = [r for r in records if r.get("strategy") == s]
            pe = [parse_numeric(r.get("energy_prefill_j")) for r in s_recs if parse_numeric(r.get("energy_prefill_j")) is not None]
            de = [parse_numeric(r.get("energy_decode_j")) for r in s_recs if parse_numeric(r.get("energy_decode_j")) is not None]
            te = [parse_numeric(r.get("energy_total_j")) for r in s_recs if parse_numeric(r.get("energy_total_j")) is not None]
            pref_means.append(np.mean(pe) if pe else 0.0)
            dec_means.append(np.mean(de) if de else 0.0)
            tot_means.append(np.mean(te) if te else 0.0)

        p1 = ax.bar(x, pref_means, width, label="Prompt Processing (Prefill)", color="#0072B2", edgecolor="black", linewidth=0.8)
        p2 = ax.bar(x, dec_means, width, bottom=pref_means, label="Generation (Decode)", color="#D55E00", edgecolor="black", linewidth=0.8)

        for i, (tot, pref, dec) in enumerate(zip(tot_means, pref_means, dec_means)):
            ax.annotate(f"{tot:.1f} J", (i, tot), ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=9.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(strat_disp, fontsize=9.5)
        ax.set_ylabel("Inference Energy (Joules)")
        ax.set_title("Prompt Processing (Prefill) vs. Autoregressive (Decode) Energy Breakdown", fontsize=12, fontweight="semibold")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        save_plot(fig, plots_dir, "02_strategy_energy_breakdown")

    # 3. Figure: Latency & TTFT Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    model_labels = []
    ttft_means = []
    latency_means = []

    for m in models:
        m_recs = [r for r in records if r.get("model") == m]
        ttft_vals = [parse_numeric(r.get("ttft_ms") or (r.get("latency_metrics", {}).get("ttft_ms") if isinstance(r.get("latency_metrics"), dict) else None)) for r in m_recs]
        ttft_vals = [t for t in ttft_vals if t is not None]
        tot_vals = [parse_numeric(r.get("total_latency_ms") or (r.get("latency_metrics", {}).get("total_latency_ms") if isinstance(r.get("latency_metrics"), dict) else None)) for r in m_recs]
        tot_vals = [t for t in tot_vals if t is not None]

        if tot_vals:
            model_labels.append(m)
            ttft_means.append(np.mean(ttft_vals) if ttft_vals else 0.0)
            latency_means.append(np.mean(tot_vals))

    if model_labels:
        y_pos = np.arange(len(model_labels))
        ax1.barh(y_pos, ttft_means, color="#56B4E9", edgecolor="black", alpha=0.85)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_labels)
        ax1.set_xlabel("Time-To-First-Token (ms)")
        ax1.set_title("Prefill Latency (TTFT)", fontsize=11, fontweight="semibold")
        ax1.grid(True, axis="x", linestyle=":", alpha=0.6)

        ax2.barh(y_pos, latency_means, color="#0072B2", edgecolor="black", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.set_xlabel("Total Generation Latency (ms)")
        ax2.set_title("End-to-End Latency", fontsize=11, fontweight="semibold")
        ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

        plt.suptitle("Cross-Model Latency & Responsiveness Breakdown", fontsize=12, fontweight="bold")
        save_plot(fig, plots_dir, "03_latency_ttft_breakdown")

    # 4. Figure: Unit Energy Cost per Token (mJ / token)
    if len(strategies) > 0:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        x = np.arange(len(strategies))
        width = 0.38

        pref_mj_list = []
        dec_mj_list = []

        for s in strategies:
            s_recs = [r for r in records if r.get("strategy") == s]
            pe = sum([parse_numeric(r.get("energy_prefill_j")) or 0.0 for r in s_recs])
            de = sum([parse_numeric(r.get("energy_decode_j")) or 0.0 for r in s_recs])
            it = sum([parse_numeric(r.get("input_tokens") or r.get("input_token_count")) or 0.0 for r in s_recs])
            ot = sum([parse_numeric(r.get("output_tokens") or r.get("output_token_count")) or 0.0 for r in s_recs])

            pref_mj_list.append((pe / it * 1000.0) if (pe > 0 and it > 0) else 0.0)
            dec_mj_list.append((de / ot * 1000.0) if (de > 0 and ot > 0) else 0.0)

        b1 = ax.bar(x - width/2, pref_mj_list, width, label="Prefill (mJ / Input Token)", color="#0072B2", edgecolor="black", linewidth=0.8)
        b2 = ax.bar(x + width/2, dec_mj_list, width, label="Decode (mJ / Output Token)", color="#D55E00", edgecolor="black", linewidth=0.8)

        for p in b1.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width()/2., h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)
        for p in b2.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width()/2., h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=9.5)
        ax.set_ylabel("Unit Energy Cost (mJ / token)")
        ax.set_title("Unit Energy Efficiency: Input Prefill vs. Output Decoding", fontsize=12, fontweight="semibold")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        save_plot(fig, plots_dir, "04_unit_energy_per_token")

    # 5. Figure: Context Length Scaling (if context_scaling runs present)
    ctx_recs = [r for r in records if "ctx_" in str(r.get("strategy")) or "context" in str(r.get("experiment_name"))]
    if ctx_recs:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for idx, m in enumerate(models):
            m_ctx = [r for r in ctx_recs if r.get("model") == m]
            lengths = []
            energies = []
            for r in m_ctx:
                strat = str(r.get("strategy", ""))
                e_val = parse_numeric(r.get("energy_total_j"))
                if "ctx_" in strat and e_val is not None:
                    try:
                        l = int(strat.replace("ctx_", ""))
                        lengths.append(l)
                        energies.append(e_val)
                    except ValueError:
                        pass
            if lengths:
                unique_l = sorted(list(set(lengths)))
                mean_e = [np.mean([energies[i] for i in range(len(lengths)) if lengths[i] == ul]) for ul in unique_l]
                ax.plot(unique_l, mean_e, marker="o", label=m, color=PALETTE[idx % len(PALETTE)], linewidth=2)

        ax.set_xlabel("Context Length (Tokens)")
        ax.set_ylabel("Total Energy (Joules)")
        ax.set_title("Context-Length vs. Energy Scaling Curves", fontsize=12, fontweight="semibold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        save_plot(fig, plots_dir, "05_context_scaling_curves")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Result Comparison Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="List of result files (.jsonl) or run directories to compare. If omitted, launches interactive selection."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_report",
        help="Directory to save generated comparison tables and figures"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive selection mode in terminal"
    )

    args = parser.parse_args()

    input_paths = args.inputs
    if not input_paths or args.interactive:
        discovered = discover_available_runs("results")
        input_paths = interactive_selection(discovered)

    if not input_paths:
        print("[EXIT] No input files selected. Exiting.")
        sys.exit(0)

    print("\n" + "=" * 75)
    print("PROMPTENERGY-BENCH: COMPARISON SUITE EXECUTION")
    print("=" * 75)
    print(f"Output Report Directory : {args.output_dir}")
    print(f"Input Run Files/Paths   : {len(input_paths)} source(s)")
    print("=" * 75)

    records = load_records(input_paths)
    if not records:
        print(f"[ERROR] No valid evaluation records found in {input_paths}. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(records):,} total evaluation records across {len(set(r.get('model') for r in records))} model(s).")

    # Generate Deliverables
    print("\n[1/2] Generating Comprehensive Comparative Tables (CSV, MD, LaTeX)...")
    generate_comparative_tables(records, args.output_dir)

    print("\n[2/2] Generating High-Resolution Publication Figures (PNG @ 300 DPI, PDF, SVG)...")
    generate_comparative_figures(records, args.output_dir)

    print("\n" + "=" * 75)
    print("COMPARISON REPORT GENERATION COMPLETE")
    print("=" * 75)
    print(f"Tables saved to  : {os.path.abspath(os.path.join(args.output_dir, 'tables'))}/")
    print(f"Figures saved to : {os.path.abspath(os.path.join(args.output_dir, 'plots'))}/")
    print("=" * 75)


if __name__ == "__main__":
    main()
