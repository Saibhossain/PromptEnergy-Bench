#!/usr/bin/env python3
"""Cross-Hardware Comparison Suite for PromptEnergy-Bench.

Ingests results from multiple hardware runs (e.g. MacBook Air M1, Windows 10-core PC, Linux CUDA)
and generates:
1. 7 Comprehensive Comparative Tables (CSV, Markdown, LaTeX)
   - Table 1: Complete Hardware Summary (hardware_summary.csv/.md/.tex)
   - Table 2: Prompt Strategy by Hardware (prompt_strategy_by_hardware.csv/.md/.tex)
   - Table 3: Model by Hardware (model_by_hardware.csv/.md/.tex)
   - Table 4: Energy-Accuracy Comparison (energy_accuracy_comparison.csv/.md/.tex)
   - Table 5: Latency Comparison (latency_comparison.csv/.md/.tex)
   - Table 6: Measurement Method Comparison (measurement_method_comparison.csv/.md/.tex)
   - Table 7: Cross-Hardware Energy Ratios (cross_hardware_energy_ratios.csv/.md/.tex)
2. 10 Publication Cross-Hardware Figures (PNG @ 300 DPI, PDF)
   - 1. energy_by_hardware_prompt (.png/.pdf)
   - 2. accuracy_by_hardware_prompt (.png/.pdf)
   - 3. ttft_by_hardware_prompt (.png/.pdf)
   - 4. latency_by_hardware_prompt (.png/.pdf)
   - 5. energy_accuracy_by_hardware (.png/.pdf)
   - 6. model_energy_across_hardware (.png/.pdf)
   - 7. context_scaling_across_hardware (.png/.pdf)
   - 8. rag_energy_across_hardware (.png/.pdf)
   - 9. hardware_energy_ratio_plot (.png/.pdf)
   - 10. summary_heatmap (.png/.pdf)

Usage:
  python scripts/compare_hardware_results.py --input-dirs results/macbook_air_m1 results/windows_10-core_pc
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def load_hardware_records(input_dirs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Loads all records grouped by normalized hardware identifier."""
    hw_records: Dict[str, List[Dict[str, Any]]] = {}

    for path in input_dirs:
        if not os.path.exists(path):
            continue
        for root, _, files in os.walk(path):
            if "hardware_comparison" in root or "analysis" in root:
                continue
            for f in files:
                if f in ("results.jsonl", "raw_results.jsonl"):
                    fpath = os.path.join(root, f)
                    try:
                        # Attempt to load metadata.json in same dir to discover device name
                        meta_device = None
                        meta_path = os.path.join(root, "metadata.json")
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, "r", encoding="utf-8") as mf:
                                    mdata = json.load(mf)
                                    dev_val = mdata.get("device")
                                    if isinstance(dev_val, dict):
                                        meta_device = dev_val.get("normalized_name")
                                    elif isinstance(dev_val, str):
                                        meta_device = dev_val
                            except Exception:
                                pass

                        with open(fpath, "r", encoding="utf-8") as handle:
                            for line in handle:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    rec = json.loads(line)
                                    hw = rec.get("hardware_identifier") or rec.get("device") or meta_device
                                    if not hw:
                                        norm_root = root.replace("\\", "/").lower()
                                        if "macbook" in norm_root or "m1" in norm_root:
                                            hw = "macbook_air_m1"
                                        elif "windows" in norm_root or "10core" in norm_root:
                                            hw = "windows_10core_pc"
                                        else:
                                            hw = "Unknown Device"

                                    # Normalize naming
                                    hw_str = str(hw).lower()
                                    if "m1" in hw_str or "mac" in hw_str:
                                        hw_name = "MacBook Air M1"
                                    elif "windows" in hw_str or "10core" in hw_str or "10-core" in hw_str:
                                        hw_name = "Windows 10-Core PC"
                                    else:
                                        hw_name = str(hw)

                                    hw_records.setdefault(hw_name, []).append(rec)
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        print(f"Warning: error reading {fpath}: {e}")
    return hw_records


def save_table_bundle(
    headers: List[str],
    rows: List[List[Any]],
    output_dir: str,
    base_name: str,
    title: str = ""
):
    """Saves a table as CSV, Markdown (.md), and LaTeX (.tex)."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # 2. Markdown
    md_path = os.path.join(output_dir, f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---:"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")

    # 3. LaTeX
    tex_path = os.path.join(output_dir, f"{base_name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        col_align = "l" + "r" * (len(headers) - 1)
        f.write(f"\\begin{{tabular}}{{{col_align}}}\n\\hline\n")
        f.write(" & ".join([h.replace("_", "\\_").replace("%", "\\%") for h in headers]) + " \\\\\n\\hline\n")
        for row in rows:
            clean_row = [str(cell).replace("_", "\\_").replace("%", "\\%") for cell in row]
            f.write(" & ".join(clean_row) + " \\\\\n")
        f.write("\\hline\n\\end{{tabular}}\n")

    print(f"  [TABLE] {base_name} (CSV, MD, LaTeX)")


def generate_all_comparison_tables(hw_records: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Generates all 7 required comparative summary tables."""

    # Table 1: Hardware Summary
    headers_1 = [
        "Hardware", "Model", "Prompt Strategy", "Accuracy (%)", "Mean Energy (J)",
        "Median Energy (J)", "Energy SD", "Mean TTFT (ms)", "Mean Latency (ms)",
        "Input Tokens", "Output Tokens", "Measurement Method", "Runs"
    ]
    rows_1 = []

    # Table 2: Prompt Strategy by Hardware
    headers_2 = ["Prompt Strategy", "Hardware", "Accuracy (%)", "Mean Energy (J)", "Mean TTFT (ms)", "Mean Latency (ms)", "Runs"]
    rows_2 = []

    # Table 3: Model by Hardware
    headers_3 = ["Model", "Hardware", "Accuracy (%)", "Mean Energy (J)", "Mean Latency (ms)", "Throughput (tok/s)", "Runs"]
    rows_3 = []

    # Table 4: Energy-Accuracy Comparison
    headers_4 = ["Hardware", "Model", "Prompt Strategy", "Accuracy (%)", "Mean Energy (J)", "Energy / Correct (J)", "Accuracy / Joule"]
    rows_4 = []

    # Table 5: Latency Comparison
    headers_5 = ["Hardware", "Prompt Strategy", "Mean TTFT (ms)", "Mean Decode Latency (ms)", "Mean Total Latency (ms)"]
    rows_5 = []

    # Table 6: Measurement Method Comparison
    headers_6 = ["Hardware", "Measurement Method", "Quality Level", "Measurement Level", "Total Samples Measured"]
    rows_6 = []

    for hw, records in sorted(hw_records.items()):
        strat_groups: Dict[str, List[Dict[str, Any]]] = {}
        model_groups: Dict[str, List[Dict[str, Any]]] = {}
        method_groups: Dict[Tuple[str, str, str], int] = {}

        for r in records:
            m = r.get("model") or r.get("model_name") or "unknown"
            s = r.get("strategy") or r.get("prompt_strategy") or "unknown"
            strat_groups.setdefault(s, []).append(r)
            model_groups.setdefault(m, []).append(r)

            meth = r.get("energy_measurement_method", "software_estimated")
            qual = r.get("energy_quality", "software_estimate")
            lvl = r.get("energy_measurement_level", "estimated_system")
            method_groups[(meth, qual, lvl)] = method_groups.get((meth, qual, lvl), 0) + 1

        for (meth, qual, lvl), count in method_groups.items():
            rows_6.append([hw, meth, qual, lvl, count])

        for strat, recs in sorted(strat_groups.items()):
            energies = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
            ttfts = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms")]
            lats = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
            in_toks = [float(r["input_tokens"]) for r in recs if r.get("input_tokens") is not None]
            out_toks = [float(r["output_tokens"]) for r in recs if r.get("output_tokens") is not None]
            accs = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
            method = recs[0].get("energy_measurement_method", "software_estimated")
            model = recs[0].get("model") or recs[0].get("model_name", "unknown")

            mean_e = np.mean(energies) if energies else 0.0
            acc_pct = np.mean(accs) * 100.0 if accs else 0.0
            corr_count = sum(accs)
            tot_e = sum(energies)
            epc = (tot_e / corr_count) if corr_count > 0 else "N/A"
            apj = (np.mean(accs) / mean_e) if mean_e > 0 else 0.0

            rows_1.append([
                hw, model, strat, f"{acc_pct:.1f}",
                f"{mean_e:.2f}", f"{np.median(energies):.2f}" if energies else "N/A",
                f"{np.std(energies):.2f}" if len(energies) > 1 else "0.00",
                f"{np.mean(ttfts):.1f}" if ttfts else "N/A",
                f"{np.mean(lats):.1f}" if lats else "N/A",
                f"{np.mean(in_toks):.0f}" if in_toks else "N/A",
                f"{np.mean(out_toks):.0f}" if out_toks else "N/A",
                method, len(recs)
            ])

            rows_2.append([
                strat, hw, f"{acc_pct:.1f}", f"{mean_e:.2f}",
                f"{np.mean(ttfts):.1f}" if ttfts else "N/A",
                f"{np.mean(lats):.1f}" if lats else "N/A",
                len(recs)
            ])

            rows_4.append([
                hw, model, strat, f"{acc_pct:.1f}", f"{mean_e:.2f}",
                f"{epc:.2f}" if isinstance(epc, float) else epc,
                f"{apj:.6f}"
            ])

            mean_ttft_val = np.mean(ttfts) if ttfts else 0.0
            mean_lat_val = np.mean(lats) if lats else 0.0
            mean_dec = max(0.0, mean_lat_val - mean_ttft_val)
            rows_5.append([hw, strat, f"{mean_ttft_val:.1f}", f"{mean_dec:.1f}", f"{mean_lat_val:.1f}"])

        for model, recs in sorted(model_groups.items()):
            energies = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
            lats = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
            out_toks = [float(r["output_tokens"]) for r in recs if r.get("output_tokens") is not None]
            accs = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
            tp = [(out_toks[i] / (lats[i]/1000.0)) for i in range(min(len(out_toks), len(lats))) if lats[i] > 0]

            rows_3.append([
                model, hw,
                f"{np.mean(accs)*100:.1f}" if accs else "N/A",
                f"{np.mean(energies):.2f}" if energies else "N/A",
                f"{np.mean(lats):.1f}" if lats else "N/A",
                f"{np.mean(tp):.2f}" if tp else "N/A",
                len(recs)
            ])

    save_table_bundle(headers_1, rows_1, output_dir, "hardware_summary", "Cross-Hardware Experimental Summary")
    save_table_bundle(headers_2, rows_2, output_dir, "prompt_strategy_by_hardware", "Prompt Strategy Performance by Hardware")
    save_table_bundle(headers_3, rows_3, output_dir, "model_by_hardware", "Model Evaluation Performance by Hardware")
    save_table_bundle(headers_4, rows_4, output_dir, "energy_accuracy_comparison", "Cross-Hardware Energy-Accuracy Trade-off Matrix")
    save_table_bundle(headers_5, rows_5, output_dir, "latency_comparison", "Cross-Hardware Latency and Phase Decomposition")
    save_table_bundle(headers_6, rows_6, output_dir, "measurement_method_comparison", "Energy Measurement Methods Across Hardware")

    # Table 7: Cross-Hardware Energy Ratios
    generate_cross_hardware_energy_ratios(hw_records, output_dir)


def generate_cross_hardware_energy_ratios(hw_records: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Generates Table 7: Cross-hardware Energy Ratios."""
    headers = [
        "Model", "Prompt Strategy", "Hardware A", "Hardware B",
        "Energy A (J)", "Energy B (J)", "Energy Ratio (A / B)", "Comparability Status"
    ]
    rows = []

    hw_list = sorted(hw_records.keys())
    if len(hw_list) >= 2:
        hw_a = hw_list[0]
        hw_b = hw_list[1]

        map_a: Dict[Tuple[str, str], List[float]] = {}
        for r in hw_records[hw_a]:
            m = r.get("model") or r.get("model_name")
            s = r.get("strategy") or r.get("prompt_strategy")
            e = r.get("energy_total_j")
            if m and s and e:
                map_a.setdefault((m, s), []).append(float(e))

        map_b: Dict[Tuple[str, str], List[float]] = {}
        for r in hw_records[hw_b]:
            m = r.get("model") or r.get("model_name")
            s = r.get("strategy") or r.get("prompt_strategy")
            e = r.get("energy_total_j")
            if m and s and e:
                map_b.setdefault((m, s), []).append(float(e))

        common_keys = sorted(set(map_a.keys()).intersection(set(map_b.keys())))
        for (m, s) in common_keys:
            e_a = np.mean(map_a[(m, s)])
            e_b = np.mean(map_b[(m, s)])
            ratio = (e_a / e_b) if e_b > 0 else 0.0

            rows.append([
                m, s, hw_a, hw_b,
                f"{e_a:.2f}", f"{e_b:.2f}", f"{ratio:.2f}",
                "Valid (Identical Model & Strategy)"
            ])

    save_table_bundle(headers, rows, output_dir, "cross_hardware_energy_ratios", "Cross-Hardware Energy Ratios")


def save_comp_fig(fig: plt.Figure, output_dir: str, base_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED FIG] {base_name}.png / .pdf")


def generate_all_comparison_figures(hw_records: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Generates all 10 required Cross-Hardware figures in PNG & PDF."""
    hw_names = sorted(hw_records.keys())
    strats = ["zero_shot_direct", "few_shot_3", "short_cot", "zero_shot_cot", "long_cot"]

    # 1. Energy by hardware and prompt strategy
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strats))
    width = 0.35
    for i, hw in enumerate(hw_names):
        vals = []
        for s in strats:
            e_list = [float(r["energy_total_j"]) for r in hw_records[hw]
                      if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("energy_total_j")]
            vals.append(np.mean(e_list) if e_list else 0.0)
        offset = (i - (len(hw_names)-1)/2) * width
        ax1.bar(x + offset, vals, width, label=hw, color=PALETTE[i % len(PALETTE)], edgecolor="black")
    ax1.set_ylabel("Mean Energy (Joules)")
    ax1.set_title("Energy Consumption by Hardware and Prompt Strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strats, rotation=20, ha="right")
    ax1.legend()
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig1, output_dir, "energy_by_hardware_prompt")

    # 2. Accuracy by hardware and prompt strategy
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, hw in enumerate(hw_names):
        vals = []
        for s in strats:
            acc_list = [1.0 if r.get("answer_correct") is True else 0.0 for r in hw_records[hw]
                        if (r.get("strategy") == s or r.get("prompt_strategy") == s)]
            vals.append(np.mean(acc_list)*100.0 if acc_list else 0.0)
        offset = (i - (len(hw_names)-1)/2) * width
        ax2.bar(x + offset, vals, width, label=hw, color=PALETTE[i % len(PALETTE)], edgecolor="black")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy by Hardware and Prompt Strategy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strats, rotation=20, ha="right")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig2, output_dir, "accuracy_by_hardware_prompt")

    # 3. TTFT by hardware and prompt strategy
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for i, hw in enumerate(hw_names):
        vals = []
        for s in strats:
            ttft_list = [float(r["ttft_ms"]) for r in hw_records[hw]
                         if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("ttft_ms")]
            vals.append(np.mean(ttft_list) if ttft_list else 0.0)
        offset = (i - (len(hw_names)-1)/2) * width
        ax3.bar(x + offset, vals, width, label=hw, color=PALETTE[i % len(PALETTE)], edgecolor="black")
    ax3.set_ylabel("Time-To-First-Token (TTFT, ms)")
    ax3.set_title("Time-To-First-Token by Hardware and Prompt Strategy")
    ax3.set_xticks(x)
    ax3.set_xticklabels(strats, rotation=20, ha="right")
    ax3.legend()
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig3, output_dir, "ttft_by_hardware_prompt")

    # 4. Total latency by hardware and prompt strategy
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for i, hw in enumerate(hw_names):
        vals = []
        for s in strats:
            lat_list = [float(r["total_latency_ms"]) for r in hw_records[hw]
                        if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("total_latency_ms")]
            vals.append(np.mean(lat_list) if lat_list else 0.0)
        offset = (i - (len(hw_names)-1)/2) * width
        ax4.bar(x + offset, vals, width, label=hw, color=PALETTE[i % len(PALETTE)], edgecolor="black")
    ax4.set_ylabel("Total Latency (ms)")
    ax4.set_title("Inference Latency by Hardware and Prompt Strategy")
    ax4.set_xticks(x)
    ax4.set_xticklabels(strats, rotation=20, ha="right")
    ax4.legend()
    ax4.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig4, output_dir, "latency_by_hardware_prompt")

    # 5. Energy-Accuracy Scatter plot by hardware
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    for i, hw in enumerate(hw_names):
        e_pts, acc_pts = [], []
        for r in hw_records[hw]:
            e = r.get("energy_total_j")
            acc = 1.0 if r.get("answer_correct") is True else 0.0
            if e and e > 0:
                e_pts.append(float(e))
                acc_pts.append(acc * 100.0)
        if e_pts:
            ax5.scatter(e_pts, acc_pts, label=hw, color=PALETTE[i % len(PALETTE)], alpha=0.6, edgecolors="black", s=50)
    ax5.set_xlabel("Energy (Joules)")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_title("Cross-Hardware Energy vs. Accuracy Distribution")
    ax5.legend()
    ax5.grid(True, linestyle="--", alpha=0.6)
    save_comp_fig(fig5, output_dir, "energy_accuracy_by_hardware")

    # 6. Model energy comparison across hardware
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    models = sorted(list(set(r.get("model") or r.get("model_name", "unknown") for recs in hw_records.values() for r in recs)))
    x_m = np.arange(len(models))
    for i, hw in enumerate(hw_names):
        vals = []
        for m in models:
            e_list = [float(r["energy_total_j"]) for r in hw_records[hw]
                      if (r.get("model") == m or r.get("model_name") == m) and r.get("energy_total_j")]
            vals.append(np.mean(e_list) if e_list else 0.0)
        offset = (i - (len(hw_names)-1)/2) * width
        ax6.bar(x_m + offset, vals, width, label=hw, color=PALETTE[i % len(PALETTE)], edgecolor="black")
    ax6.set_ylabel("Mean Energy (Joules)")
    ax6.set_title("Model Energy Consumption Across Hardware Platforms")
    ax6.set_xticks(x_m)
    ax6.set_xticklabels(models, rotation=20, ha="right")
    ax6.legend()
    ax6.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig6, output_dir, "model_energy_across_hardware")

    # 7. Context scaling across hardware
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    for i, hw in enumerate(hw_names):
        ctx_dict = {}
        for r in hw_records[hw]:
            strat = str(r.get("strategy", ""))
            if strat.startswith("ctx_") and r.get("energy_total_j"):
                try:
                    c_len = int(strat[4:])
                    ctx_dict.setdefault(c_len, []).append(float(r["energy_total_j"]))
                except ValueError:
                    pass
        if ctx_dict:
            cx = sorted(ctx_dict.keys())
            cy = [np.mean(ctx_dict[k]) for k in cx]
            ax7.plot(cx, cy, marker="o", linewidth=2.0, label=hw, color=PALETTE[i % len(PALETTE)])
    ax7.set_xlabel("Context Length (Tokens)")
    ax7.set_ylabel("Mean Energy (Joules)")
    ax7.set_title("Context Length Energy Scaling Across Hardware")
    ax7.legend()
    ax7.grid(True, linestyle="--", alpha=0.6)
    save_comp_fig(fig7, output_dir, "context_scaling_across_hardware")

    # 8. RAG energy across hardware
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    for i, hw in enumerate(hw_names):
        rag_energies = [float(r["energy_total_j"]) for r in hw_records[hw]
                        if ("rag" in str(r.get("strategy", "")).lower() or "rag" in str(r.get("experiment_name", "")).lower()) and r.get("energy_total_j")]
        if rag_energies:
            ax8.bar(hw, np.mean(rag_energies), color=PALETTE[i % len(PALETTE)], edgecolor="black", alpha=0.85)
    ax8.set_ylabel("Mean RAG Energy (Joules)")
    ax8.set_title("RAG Inference Energy Across Hardware Platforms")
    ax8.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig8, output_dir, "rag_energy_across_hardware")

    # 9. Hardware energy ratio plot
    fig9, ax9 = plt.subplots(figsize=(8, 5))
    if len(hw_names) >= 2:
        hw_a, hw_b = hw_names[0], hw_names[1]
        ratios = []
        r_labels = []
        for s in strats:
            ea = [float(r["energy_total_j"]) for r in hw_records[hw_a] if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("energy_total_j")]
            eb = [float(r["energy_total_j"]) for r in hw_records[hw_b] if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("energy_total_j")]
            if ea and eb:
                ratios.append(np.mean(ea) / np.mean(eb))
                r_labels.append(s)
        if ratios:
            ax9.bar(r_labels, ratios, color=PALETTE[3], edgecolor="black", alpha=0.85)
            ax9.axhline(1.0, color="red", linestyle="--", label="Parity (Ratio = 1.0)")
            ax9.set_ylabel(f"Energy Ratio ({hw_a} / {hw_b})")
            ax9.set_title(f"Cross-Hardware Energy Ratio ({hw_a} vs. {hw_b})")
            ax9.set_xticks(range(len(r_labels)))
            ax9.set_xticklabels(r_labels, rotation=20, ha="right")
            ax9.legend()
            ax9.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_comp_fig(fig9, output_dir, "hardware_energy_ratio_plot")

    # 10. Summary Heatmap
    plot_summary_heatmap(hw_records, output_dir)


def plot_summary_heatmap(hw_records: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Figure 10: Summary heatmap of energy across hardware and strategies."""
    strats = ["zero_shot_direct", "few_shot_3", "short_cot", "zero_shot_cot", "long_cot"]
    hw_names = sorted(hw_records.keys())

    matrix = np.zeros((len(hw_names), len(strats)))
    for i, hw in enumerate(hw_names):
        for j, s in enumerate(strats):
            vals = [float(r["energy_total_j"]) for r in hw_records[hw]
                    if (r.get("strategy") == s or r.get("prompt_strategy") == s) and r.get("energy_total_j")]
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(strats)))
    ax.set_yticks(np.arange(len(hw_names)))
    ax.set_xticklabels(strats, rotation=25, ha="right")
    ax.set_yticklabels(hw_names)

    for i in range(len(hw_names)):
        for j in range(len(strats)):
            ax.text(j, i, f"{matrix[i, j]:.1f} J", ha="center", va="center", color="black", fontsize=9)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Energy (Joules)", rotation=-90, va="bottom")
    ax.set_title("Cross-Hardware Energy Expenditure Heatmap")
    save_comp_fig(fig, output_dir, "summary_heatmap")


def main():
    parser = argparse.ArgumentParser(
        description="PromptEnergy-Bench Cross-Hardware Comparison Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["results"],
        help="One or more directories containing experimental hardware results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hardware_comparison",
        help="Output directory for cross-hardware tables and figures"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PROMPTENERGY-BENCH: CROSS-HARDWARE COMPARISON SUITE")
    print("=" * 70)
    print(f"Input Directories : {args.input_dirs}")
    print(f"Output Directory  : {args.output_dir}")
    print("=" * 70)

    hw_records = load_hardware_records(args.input_dirs)
    print(f"Discovered Hardware Targets: {list(hw_records.keys())}")
    for hw, recs in hw_records.items():
        print(f"  - {hw}: {len(recs)} records")

    if not hw_records:
        print("[ERROR] No hardware result records found. Run experiments first!")
        sys.exit(1)

    print("\nGenerating 7 Comparative Summary Tables (CSV, Markdown, LaTeX)...")
    generate_all_comparison_tables(hw_records, args.output_dir)

    print("\nGenerating 10 Cross-Hardware Comparison Figures (PNG & PDF)...")
    generate_all_comparison_figures(hw_records, args.output_dir)

    # Also export research summary tables in results/analysis/
    try:
        from src.analysis.tables import generate_research_summary_tables
        exported = generate_research_summary_tables("results", "results/analysis")
        print(f"\nSuccessfully generated 8 research summary CSVs in results/analysis/: {len(exported)} files.")
    except Exception as e:
        print(f"Warning: Failed to generate results/analysis tables: {e}")

    print("\n" + "=" * 70)
    print(f"Successfully generated all cross-hardware deliverables in: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
