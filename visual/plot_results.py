"""Publication-Grade Visualization Suite for PromptEnergy-Bench.

Generates 14+ publication figures (PNG @ 300 DPI, vector PDF, and SVG)
plus a diagnostic overview dashboard:
  01_accuracy_by_strategy.*
  02_energy_by_strategy.*
  02b_prefill_vs_decode_energy_stacked.*
  03_latency_by_strategy.*
  04_token_composition_by_strategy.*
  05_accuracy_energy_pareto.*
  06_accuracy_latency.*
  07_prefill_energy_vs_input_tokens.*
  08_decode_energy_vs_output_tokens.*
  09_token_energy_efficiency.*
  10_efficiency_by_strategy.*
  11_energy_distribution.*
  12_latency_distribution.*
  13_truncation_rate.*
  validation_overview.png
"""

import argparse
import datetime
import json
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.pareto import compute_pareto_frontier


# Centralized Research Plot Configuration
PLOT_CONFIG = {
    "dpi": 300,
    "font_scale": 1.15,
    "context": "paper",
    "style": "whitegrid",
    "palette": "colorblind"
}

# Apply global styling
sns.set_theme(
    context=PLOT_CONFIG["context"],
    style=PLOT_CONFIG["style"],
    font_scale=PLOT_CONFIG["font_scale"],
    palette=PLOT_CONFIG["palette"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.titlesize": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "figure.autolayout": False
})

STRATEGY_ORDER = [
    "zero_shot_direct",
    "few_shot_3",
    "zero_shot_cot",
    "short_cot",
    "long_cot",
    "ctx_0",
    "ctx_512",
    "ctx_1024",
    "ctx_2048",
    "ctx_4096",
    "ctx_8192",
    "rag_top_1",
    "rag_top_3",
    "rag_top_5"
]

STRATEGY_LABELS = {
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


def get_strategy_label(strat: str) -> str:
    if strat in STRATEGY_LABELS:
        return STRATEGY_LABELS[strat]
    if str(strat).startswith("ctx_"):
        return f"Context {strat[4:]} tokens"
    if str(strat).startswith("rag_top_"):
        return f"BM25 RAG (top-{strat[8:]})"
    return str(strat).replace("_", " ").title()


def get_palette_for_items(items: List[Any]) -> Dict[str, Any]:
    """Dynamically builds a color mapping for a list of items."""
    unique_items = [str(x) for x in list(dict.fromkeys(items))]
    colors = sns.color_palette("colorblind", n_colors=max(len(unique_items), 1))
    return dict(zip(unique_items, colors))


def save_publication_figure(fig: plt.Figure, output_dir: str, stem: str) -> List[str]:
    """Saves figure in high-res PNG (300 DPI), vector PDF, and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for ext in ["png", "pdf", "svg"]:
        fpath = os.path.join(output_dir, f"{stem}.{ext}")
        kwargs = {"dpi": PLOT_CONFIG["dpi"], "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        fig.savefig(fpath, **kwargs)
        saved.append(fpath)
    plt.close(fig)
    return saved


def smart_annotate_scatter(
    ax: plt.Axes,
    points: List[Tuple[float, float, str, bool]],
    x_range_pad: float = 0.05,
    y_range_pad: float = 0.05
) -> None:
    """Smart collision-avoiding annotation for 2D scatter plots.
    
    Args:
        points: List of (x, y, label, is_highlighted)
    """
    if not points:
        return

    # Coordinate sorting and clustering
    sorted_pts = sorted(points, key=lambda p: (p[1], p[0]))
    
    # Pre-defined radial offset positions
    offset_cycle = [
        (10, 8),     # Top-Right
        (-12, 14),   # Top-Left
        (12, -14),   # Bottom-Right
        (-14, -14),  # Bottom-Left
        (0, 18),     # High Above
        (0, -22),    # Low Below
        (18, 0),     # Far Right
        (-20, 0)     # Far Left
    ]

    used_boxes = []

    for idx, (x, y, label, is_high) in enumerate(sorted_pts):
        offset = offset_cycle[idx % len(offset_cycle)]
        
        # Adjust if point is near zero accuracy
        if y < 1.0:
            # Alternating vertical offsets above zero line
            alt_y = 10 + ((idx % 3) * 14)
            offset = (offset[0], alt_y)

        fontweight = "bold" if is_high else "normal"
        fontsize = 9.0 if is_high else 8.5
        edgecolor = "#E64B35" if is_high else "#CCCCCC"
        linewidth = 1.0 if is_high else 0.6
        bg_alpha = 0.95 if is_high else 0.88

        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=fontsize,
            fontweight=fontweight,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=bg_alpha,
                edgecolor=edgecolor,
                linewidth=linewidth
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                linewidth=0.6,
                alpha=0.6
            ) if abs(offset[0]) > 14 or abs(offset[1]) > 14 else None,
            zorder=6
        )


def load_and_filter_results(
    results_path: str,
    model_filter: Optional[str] = None,
    device_filter: Optional[str] = None,
    run_id_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads results.jsonl and returns (all_records_df, successful_records_df)."""
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    df_all = pd.DataFrame(records)
    if df_all.empty:
        return df_all, df_all

    # Optional filters
    if model_filter and "model" in df_all.columns:
        df_all = df_all[df_all["model"] == model_filter].copy()
    if device_filter and "device" in df_all.columns:
        df_all = df_all[df_all["device"] == device_filter].copy()
    if run_id_filter and "run_id" in df_all.columns:
        df_all = df_all[df_all["run_id"] == run_id_filter].copy()

    # Calculate Prefill & Decode Energy if not explicitly present
    if "total_latency_ms" in df_all.columns and "energy_total_j" in df_all.columns:
        valid_mask = (df_all["total_latency_ms"] > 0) & (df_all["energy_total_j"] > 0)
        ttft_col = df_all["ttft_ms"] if "ttft_ms" in df_all.columns else 0.0
        
        if "energy_prefill_j" not in df_all.columns or df_all["energy_prefill_j"].isna().all():
            df_all["energy_prefill_j"] = np.where(
                valid_mask,
                df_all["energy_total_j"] * (ttft_col / df_all["total_latency_ms"]),
                np.nan
            )

        if "energy_decode_j" not in df_all.columns or df_all["energy_decode_j"].isna().all():
            gen_col = df_all["generation_latency_ms"] if "generation_latency_ms" in df_all.columns else (df_all["total_latency_ms"] - ttft_col)
            df_all["energy_decode_j"] = np.where(
                valid_mask,
                df_all["energy_total_j"] * (gen_col / df_all["total_latency_ms"]),
                np.nan
            )

    # Map strategy display names and categories
    if "strategy" in df_all.columns:
        df_all["strategy_display"] = df_all["strategy"].map(get_strategy_label)
        cat_order = [get_strategy_label(s) for s in STRATEGY_ORDER if s in df_all["strategy"].unique()]
        for s in df_all["strategy"].unique():
            disp = get_strategy_label(s)
            if disp not in cat_order:
                cat_order.append(disp)
        df_all["strategy_display"] = pd.Categorical(df_all["strategy_display"], categories=cat_order, ordered=True)

    df_success = df_all[df_all["status"] == "success"].copy() if "status" in df_all.columns else df_all.copy()
    return df_all, df_success


# ==============================================================================
# RESEARCH-GRADE PUBLICATION FIGURES
# ==============================================================================

def plot_01_accuracy_by_strategy(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 1: Accuracy (%) by Prompt Strategy."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    acc_df = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True)["answer_correct"].mean()
    acc_df["accuracy_pct"] = acc_df["answer_correct"] * 100.0

    palette = get_palette_for_items(acc_df["strategy_display"])
    bars = sns.barplot(
        data=acc_df,
        x="strategy_display",
        y="accuracy_pct",
        hue="strategy_display",
        palette=palette,
        legend=False,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    # Annotate bar values
    for p in bars.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=9.5, fontweight="semibold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_ylim(0, max(100, (acc_df["accuracy_pct"].max() + 18) if not acc_df.empty else 100))
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "01_accuracy_by_strategy")


def plot_02_energy_by_strategy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 2: Mean Energy per Inference (J) by Strategy."""
    if "energy_total_j" not in df_success.columns or df_success["energy_total_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    errorbar = ("ci", 95) if reps > 1 else None
    palette = get_palette_for_items(df_success["strategy_display"])
    bars = sns.barplot(
        data=df_success,
        x="strategy_display",
        y="energy_total_j",
        hue="strategy_display",
        palette=palette,
        legend=False,
        errorbar=errorbar,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    for p in bars.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f} J", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=9.5, fontweight="semibold")

    ax.set_ylabel("Mean Total Energy per Query (Joules)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "02_energy_by_strategy")


def plot_02b_prefill_vs_decode_stacked(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 2B: Prompt Processing (Prefill) vs. Autoregressive (Decode) Energy Breakdown."""
    needed = ["strategy", "strategy_display", "energy_prefill_j", "energy_decode_j"]
    if not all(c in df_success.columns for c in needed) or df_success["energy_prefill_j"].isna().all():
        return []

    agg = df_success.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "energy_prefill_j": "mean",
        "energy_decode_j": "mean",
        "energy_total_j": "mean"
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x_indices = np.arange(len(agg))
    width = 0.52

    prefill_vals = agg["energy_prefill_j"].values
    decode_vals = agg["energy_decode_j"].values
    total_vals = agg["energy_total_j"].values

    p1 = ax.bar(x_indices, prefill_vals, width, label="Prompt Processing (Prefill)", color="#0072B2", edgecolor="#333333", linewidth=0.8)
    p2 = ax.bar(x_indices, decode_vals, width, bottom=prefill_vals, label="Generation (Decode)", color="#D55E00", edgecolor="#333333", linewidth=0.8)

    # Annotations
    for i, (tot, pref, dec) in enumerate(zip(total_vals, prefill_vals, decode_vals)):
        pref_pct = (pref / tot) * 100.0 if tot > 0 else 0.0
        dec_pct = (dec / tot) * 100.0 if tot > 0 else 0.0

        # Total on top
        ax.annotate(f"{tot:.2f} J", (i, tot), ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=9.5, fontweight="bold")
        # Percent share inside bars if space allows
        if pref > 1.5:
            ax.annotate(f"{pref:.1f}J ({pref_pct:.0f}%)", (i, pref / 2.0), ha="center", va="center", color="white", fontsize=8.5, fontweight="semibold")
        if dec > 5.0:
            ax.annotate(f"{dec:.1f}J ({dec_pct:.0f}%)", (i, pref + (dec / 2.0)), ha="center", va="center", color="white", fontsize=8.5, fontweight="semibold")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(agg["strategy_display"], rotation=15, ha="right")
    ax.set_ylabel("Inference Energy (Joules)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_title("Inference Energy Decomposition: Prefill vs. Decode", fontsize=12, fontweight="semibold")
    ax.legend(loc="upper left")
    return save_publication_figure(fig, out_dir, "02b_prefill_vs_decode_energy_stacked")


def plot_03_latency_by_strategy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 3: Latency Decomposition (TTFT vs Generation Latency vs Total Latency)."""
    needed = ["strategy", "strategy_display", "total_latency_ms"]
    if not all(c in df_success.columns for c in needed) or df_success["total_latency_ms"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    agg = df_success.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "ttft_ms": "mean",
        "generation_latency_ms": "mean",
        "total_latency_ms": "mean"
    })

    melted = pd.melt(
        agg,
        id_vars=["strategy_display"],
        value_vars=["ttft_ms", "generation_latency_ms", "total_latency_ms"],
        var_name="latency_phase",
        value_name="latency_ms"
    )
    phase_labels = {
        "ttft_ms": "Time to First Token (TTFT, ms)",
        "generation_latency_ms": "Generation Latency (ms)",
        "total_latency_ms": "Total Latency (ms)"
    }
    melted["phase_display"] = melted["latency_phase"].map(phase_labels)

    sns.barplot(
        data=melted,
        x="strategy_display",
        y="latency_ms",
        hue="phase_display",
        palette=["#56B4E9", "#E69F00", "#009E73"],
        edgecolor="#333333",
        linewidth=0.8,
        ax=ax
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Prompting Strategy")
    ax.legend(title="Component", loc="upper left")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "03_latency_by_strategy")


def plot_04_token_composition_by_strategy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 4: Token Composition (Thinking Tokens vs Visible Output Tokens)."""
    if "output_tokens" not in df_success.columns:
        return []

    agg = df_success.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "thinking_tokens": lambda x: x.dropna().mean() if not x.dropna().empty else 0.0,
        "visible_output_tokens": lambda x: x.dropna().mean() if not x.dropna().empty else 0.0,
        "output_tokens": "mean"
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x_indices = np.arange(len(agg))
    width = 0.52

    th_vals = agg["thinking_tokens"].values
    vis_vals = agg["visible_output_tokens"].values

    p1 = ax.bar(x_indices, th_vals, width, label="Thinking Tokens", color="#CC79A7", edgecolor="#333333", linewidth=0.8)
    p2 = ax.bar(x_indices, vis_vals, width, bottom=th_vals, label="Visible Output Tokens", color="#0072B2", edgecolor="#333333", linewidth=0.8)

    for i, (tot, th, vis) in enumerate(zip(agg["output_tokens"], th_vals, vis_vals)):
        height = th + vis
        ax.annotate(f"{tot:.0f}", (i, height), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9.5, fontweight="semibold")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(agg["strategy_display"], rotation=15, ha="right")
    ax.set_ylabel("Mean Output Token Count")
    ax.set_xlabel("Prompting Strategy")
    ax.legend(loc="upper left")
    return save_publication_figure(fig, out_dir, "04_token_composition_by_strategy")


def plot_05_accuracy_energy_pareto(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 5: Accuracy–Energy Pareto Frontier with Collision-Free Annotations."""
    if "energy_total_j" not in df_all.columns or df_all["energy_total_j"].isna().all():
        return []

    agg = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "answer_correct": "mean",
        "energy_total_j": "mean",
        "total_latency_ms": "mean"
    })
    agg["accuracy_pct"] = agg["answer_correct"] * 100.0

    candidates = [{
        "strategy": row["strategy"],
        "accuracy": row["accuracy_pct"],
        "energy_j": row["energy_total_j"]
    } for _, row in agg.iterrows()]

    pareto_pts = compute_pareto_frontier(candidates)
    pareto_strats = {p["strategy"] for p in pareto_pts}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = get_palette_for_items(agg["strategy"])

    # Scatter of all strategies
    sns.scatterplot(
        data=agg,
        x="energy_total_j",
        y="accuracy_pct",
        hue="strategy",
        palette=palette,
        s=180,
        legend=False,
        edgecolor="#333333",
        linewidth=1.2,
        zorder=5,
        ax=ax
    )

    # Draw dashed Pareto line
    if len(pareto_pts) > 1:
        p_df = pd.DataFrame(pareto_pts).sort_values("energy_j")
        ax.plot(p_df["energy_j"], p_df["accuracy"], linestyle="--", color="#D55E00", linewidth=2.0, label="Pareto Frontier", zorder=4)

    # Smart non-overlapping annotations
    annot_points = []
    for _, row in agg.iterrows():
        strat = row["strategy"]
        is_p = strat in pareto_strats
        label = f"{get_strategy_label(strat)}{' [Pareto]' if is_p else ''}"
        annot_points.append((row["energy_total_j"], row["accuracy_pct"], label, is_p))

    smart_annotate_scatter(ax, annot_points)

    ax.set_xlabel("Mean Total Energy per Inference (Joules)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Energy Expenditure (Pareto Frontier)", fontsize=12, fontweight="semibold")
    ax.set_ylim(-3, max(100, agg["accuracy_pct"].max() + 15))
    if len(pareto_pts) > 1:
        ax.legend(loc="lower right")
    return save_publication_figure(fig, out_dir, "05_accuracy_energy_pareto")


def plot_06_accuracy_latency(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 6: Accuracy vs Latency Trade-off."""
    if "total_latency_ms" not in df_all.columns:
        return []

    agg = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "answer_correct": "mean",
        "total_latency_ms": "mean"
    })
    agg["accuracy_pct"] = agg["answer_correct"] * 100.0

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    palette = get_palette_for_items(agg["strategy"])
    sns.scatterplot(
        data=agg,
        x="total_latency_ms",
        y="accuracy_pct",
        hue="strategy",
        palette=palette,
        s=160,
        legend=False,
        edgecolor="#333333",
        linewidth=1.2,
        zorder=5,
        ax=ax
    )

    annot_points = [(row["total_latency_ms"], row["accuracy_pct"], get_strategy_label(row["strategy"]), False) for _, row in agg.iterrows()]
    smart_annotate_scatter(ax, annot_points)

    ax.set_xlabel("Mean Total Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Latency Trade-off", fontsize=12, fontweight="semibold")
    return save_publication_figure(fig, out_dir, "06_accuracy_latency")


def plot_07_prefill_energy_vs_input_tokens(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 7: Prompt Processing (Prefill) Energy vs. Input Token Load."""
    needed = ["input_tokens", "energy_prefill_j"]
    if not all(c in df_success.columns for c in needed) or df_success["energy_prefill_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    palette = get_palette_for_items(df_success["strategy"])

    sns.scatterplot(
        data=df_success,
        x="input_tokens",
        y="energy_prefill_j",
        hue="strategy",
        palette=palette,
        alpha=0.65,
        s=40,
        ax=ax
    )

    # Linear trendline fit
    valid = df_success.dropna(subset=["input_tokens", "energy_prefill_j"])
    if len(valid) > 10:
        x_vals = valid["input_tokens"].values
        y_vals = valid["energy_prefill_j"].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color="#333333", linestyle="--", linewidth=1.5,
                label=f"Fit: {slope*1000:.2f} mJ/input-token (R²={np.corrcoef(x_vals, y_vals)[0,1]**2:.2f})")

    ax.set_xlabel("Input Prompt Tokens ($L_{\\mathrm{prompt}}$)")
    ax.set_ylabel("Prefill Energy ($E_{\\mathrm{prefill}}$, Joules)")
    ax.set_title("Prompt Processing (Prefill) Energy Scaling", fontsize=12, fontweight="semibold")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_publication_figure(fig, out_dir, "07_prefill_energy_vs_input_tokens")


def plot_08_decode_energy_vs_output_tokens(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 8: Generation (Decode) Energy vs. Generated Output Tokens."""
    needed = ["output_tokens", "energy_decode_j"]
    if not all(c in df_success.columns for c in needed) or df_success["energy_decode_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    palette = get_palette_for_items(df_success["strategy"])

    sns.scatterplot(
        data=df_success,
        x="output_tokens",
        y="energy_decode_j",
        hue="strategy",
        palette=palette,
        alpha=0.65,
        s=40,
        ax=ax
    )

    valid = df_success.dropna(subset=["output_tokens", "energy_decode_j"])
    if len(valid) > 10:
        x_vals = valid["output_tokens"].values
        y_vals = valid["energy_decode_j"].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color="#D55E00", linestyle="--", linewidth=1.5,
                label=f"Fit: {slope*1000:.2f} mJ/decode-token (R²={np.corrcoef(x_vals, y_vals)[0,1]**2:.2f})")

    ax.set_xlabel("Generated Output Tokens ($L_{\\mathrm{gen}}$)")
    ax.set_ylabel("Decode Energy ($E_{\\mathrm{decode}}$, Joules)")
    ax.set_title("Autoregressive Generation (Decode) Energy Scaling", fontsize=12, fontweight="semibold")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_publication_figure(fig, out_dir, "08_decode_energy_vs_output_tokens")


def plot_09_token_energy_efficiency(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 9: Unit Energy Cost Comparison (Prefill mJ/token vs. Decode mJ/token)."""
    needed = ["strategy", "strategy_display", "input_tokens", "output_tokens", "energy_prefill_j", "energy_decode_j"]
    if not all(c in df_success.columns for c in needed):
        return []

    agg = df_success.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "input_tokens": "sum",
        "output_tokens": "sum",
        "energy_prefill_j": "sum",
        "energy_decode_j": "sum"
    })

    agg["prefill_mj_per_tok"] = (agg["energy_prefill_j"] / agg["input_tokens"]) * 1000.0
    agg["decode_mj_per_tok"] = (agg["energy_decode_j"] / agg["output_tokens"]) * 1000.0

    melted = pd.melt(
        agg,
        id_vars=["strategy_display"],
        value_vars=["prefill_mj_per_tok", "decode_mj_per_tok"],
        var_name="token_phase",
        value_name="mj_per_token"
    )
    phase_map = {
        "prefill_mj_per_tok": "Prefill (mJ / Input Token)",
        "decode_mj_per_tok": "Decode (mJ / Output Token)"
    }
    melted["phase_label"] = melted["token_phase"].map(phase_map)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = sns.barplot(
        data=melted,
        x="strategy_display",
        y="mj_per_token",
        hue="phase_label",
        palette=["#0072B2", "#D55E00"],
        edgecolor="#333333",
        linewidth=0.8,
        ax=ax
    )

    for p in bars.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=8.5)

    ax.set_ylabel("Energy Cost per Token (mJ / token)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_title("Unit Energy Efficiency: Input Prefill vs. Output Decoding", fontsize=12, fontweight="semibold")
    ax.legend(title="Phase", loc="upper left")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "09_token_energy_efficiency")


def plot_10_efficiency_by_strategy(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 10: Strategy Efficiency (Accuracy per Joule)."""
    if "energy_total_j" not in df_all.columns:
        return []

    agg = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "answer_correct": "mean",
        "energy_total_j": "mean"
    })
    agg["accuracy_per_joule"] = agg.apply(
        lambda r: (r["answer_correct"] / r["energy_total_j"]) if r["energy_total_j"] > 0 else 0.0,
        axis=1
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = get_palette_for_items(agg["strategy_display"])
    bars = sns.barplot(
        data=agg,
        x="strategy_display",
        y="accuracy_per_joule",
        hue="strategy_display",
        palette=palette,
        legend=False,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    for p in bars.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

    ax.set_ylabel("Efficiency (Accuracy / Joule)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_title("Accuracy per Joule ($APJ$) Across Prompting Strategies", fontsize=12, fontweight="semibold")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "10_efficiency_by_strategy")


def plot_11_energy_distribution(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 11: Distribution of Energy."""
    if "energy_total_j" not in df_success.columns or df_success["energy_total_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = get_palette_for_items(df_success["strategy_display"])

    if reps > 1:
        sns.boxplot(data=df_success, x="strategy_display", y="energy_total_j", hue="strategy_display", palette=palette, legend=False, ax=ax, width=0.4, fliersize=2)
        sns.stripplot(data=df_success, x="strategy_display", y="energy_total_j", color="#222222", alpha=0.3, size=3, jitter=0.2, ax=ax)
    else:
        sns.stripplot(data=df_success, x="strategy_display", y="energy_total_j", hue="strategy_display", palette=palette, legend=False, size=4, jitter=0.25, alpha=0.6, ax=ax)

    ax.set_ylabel("Inference Energy (Joules)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "11_energy_distribution")


def plot_12_latency_distribution(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 12: Distribution of Latency."""
    if "total_latency_ms" not in df_success.columns or df_success["total_latency_ms"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = get_palette_for_items(df_success["strategy_display"])

    if reps > 1:
        sns.boxplot(data=df_success, x="strategy_display", y="total_latency_ms", hue="strategy_display", palette=palette, legend=False, ax=ax, width=0.4, fliersize=2)
        sns.stripplot(data=df_success, x="strategy_display", y="total_latency_ms", color="#222222", alpha=0.3, size=3, jitter=0.2, ax=ax)
    else:
        sns.stripplot(data=df_success, x="strategy_display", y="total_latency_ms", hue="strategy_display", palette=palette, legend=False, size=4, jitter=0.25, alpha=0.6, ax=ax)

    ax.set_ylabel("Total Latency (ms)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "12_latency_distribution")


def plot_13_truncation_rate(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 13: Generation Truncation Rate (%) by Strategy."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    trunc_df = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "generation_truncated": lambda x: (sum(1 for v in x if v is True) / len(x)) * 100.0 if len(x) > 0 else 0.0
    })

    palette = get_palette_for_items(trunc_df["strategy_display"])
    bars = sns.barplot(
        data=trunc_df,
        x="strategy_display",
        y="generation_truncated",
        hue="strategy_display",
        palette=palette,
        legend=False,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    for p in bars.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., max(0.5, height)),
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=9.5, fontweight="semibold")

    ax.set_ylabel("Truncation Rate (%)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_ylim(0, max(10, trunc_df["generation_truncated"].max() + 15))
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "13_truncation_rate")


def plot_validation_overview(df_all: pd.DataFrame, summary: Dict[str, Any], out_dir: str) -> str:
    """Creates a concise, publication-style multi-panel validation diagnostic dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8))
    fig.suptitle("PromptEnergy-Bench Experiment Validation Dashboard", fontsize=14, fontweight="bold", y=0.98)

    # 1. Accuracy Panel
    acc_df = df_all.groupby("strategy_display", observed=True)["answer_correct"].mean().reset_index()
    acc_df["acc_pct"] = acc_df["answer_correct"] * 100.0
    sns.barplot(data=acc_df, x="strategy_display", y="acc_pct", hue="strategy_display", legend=False, ax=axes[0, 0], palette="crest")
    axes[0, 0].set_title("Accuracy (%)", fontsize=11, fontweight="semibold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=25)

    # 2. Truncation Rate Panel
    trunc_df = df_all.groupby("strategy_display", observed=True)["generation_truncated"].apply(
        lambda s: (sum(1 for v in s if v is True) / len(s)) * 100.0 if len(s) > 0 else 0.0
    ).reset_index()
    sns.barplot(data=trunc_df, x="strategy_display", y="generation_truncated", hue="strategy_display", legend=False, ax=axes[0, 1], palette="Reds_r")
    axes[0, 1].set_title("Truncation Rate (%)", fontsize=11, fontweight="semibold")
    axes[0, 1].set_ylabel("Truncated (%)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=25)

    # 3. Energy Consumption Panel
    if "energy_total_j" in df_all.columns:
        e_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=e_df, x="strategy_display", y="energy_total_j", hue="strategy_display", legend=False, ax=axes[0, 2], palette="viridis")
        axes[0, 2].set_title("Mean Total Energy (J)", fontsize=11, fontweight="semibold")
        axes[0, 2].set_ylabel("Energy (Joules)")
        axes[0, 2].set_xlabel("")
        axes[0, 2].tick_params(axis="x", rotation=25)

    # 4. Latency Panel
    if "total_latency_ms" in df_all.columns:
        l_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=l_df, x="strategy_display", y="total_latency_ms", hue="strategy_display", legend=False, ax=axes[1, 0], palette="mako")
        axes[1, 0].set_title("Mean Latency (ms)", fontsize=11, fontweight="semibold")
        axes[1, 0].set_ylabel("Total Latency (ms)")
        axes[1, 0].set_xlabel("")
        axes[1, 0].tick_params(axis="x", rotation=25)

    # 5. Output Tokens Panel
    if "output_tokens" in df_all.columns:
        tok_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=tok_df, x="strategy_display", y="output_tokens", hue="strategy_display", legend=False, ax=axes[1, 1], palette="flare")
        axes[1, 1].set_title("Mean Generated Tokens", fontsize=11, fontweight="semibold")
        axes[1, 1].set_ylabel("Tokens")
        axes[1, 1].set_xlabel("")
        axes[1, 1].tick_params(axis="x", rotation=25)

    # 6. Diagnostic Summary Card
    metrics = summary.get("metrics", {})
    diag_text = (
        f"Validation Scope:\n"
        f" • Dataset: GSM8K (TEST split)\n"
        f" • Evaluation Examples: {summary.get('evaluation_examples', 50)}\n"
        f" • Total Inferences: {metrics.get('requested_inference_runs', len(df_all))}\n"
        f" • Successful Runs: {metrics.get('successful_inference_runs', len(df_all))}\n\n"
        f"Overall Outcome:\n"
        f" • Accuracy: {metrics.get('accuracy', 0.0) * 100.0:.2f}%\n"
        f" • Truncation Rate: {metrics.get('truncation_rate', 0.0) * 100.0:.2f}%\n"
        f" • Mean Prefill Energy: {metrics.get('mean_prefill_energy_j', 'N/A')} J\n"
        f" • Mean Decode Energy: {metrics.get('mean_decode_energy_j', 'N/A')} J\n"
        f" • Mean Total Energy: {metrics.get('mean_energy_j', 'N/A')} J\n"
        f" • Mean Latency: {metrics.get('mean_total_latency_ms', 'N/A')} ms"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.95, diag_text, transform=axes[1, 2].transAxes,
                    fontsize=10.0, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.7", facecolor="#F8F9FA", edgecolor="#CCCCCC"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(out_dir, "validation_overview.png")
    fig.savefig(out_file, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    return out_file


def save_plot_metadata(
    output_dir: str,
    run_id: str,
    source_results_file: str,
    df_all: pd.DataFrame,
    reps: int
) -> str:
    """Saves plot_metadata.json inside plots directory."""
    meta = {
        "run_id": run_id,
        "source_results_file": os.path.abspath(source_results_file),
        "plot_generation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": list(df_all["model"].unique()) if "model" in df_all.columns else [],
        "devices": list(df_all["device"].unique()) if "device" in df_all.columns else [],
        "strategies": list(df_all["strategy"].unique()) if "strategy" in df_all.columns else [],
        "sample_count": len(df_all),
        "repetitions": reps,
        "plot_version": "2.0-research",
        "plot_config": PLOT_CONFIG
    }
    meta_path = os.path.join(output_dir, "plot_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


def generate_all_plots_for_run(
    target_dir: str,
    results_file: Optional[str] = None,
    model_filter: Optional[str] = None,
    device_filter: Optional[str] = None,
    run_id_filter: Optional[str] = None
) -> List[str]:
    """Generates all research figures + diagnostic overview and metadata for a run."""
    if not results_file:
        results_file = os.path.join(target_dir, "results.jsonl")

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return []

    df_all, df_success = load_and_filter_results(
        results_file,
        model_filter=model_filter,
        device_filter=device_filter,
        run_id_filter=run_id_filter
    )

    if df_all.empty:
        print("No matching result records found to plot.")
        return []

    # Load summary and config if available
    summary = {}
    config = {}
    s_path = os.path.join(target_dir, "summary.json")
    c_path = os.path.join(target_dir, "config.json")
    if os.path.exists(s_path):
        with open(s_path, "r") as f:
            summary = json.load(f)
    if os.path.exists(c_path):
        with open(c_path, "r") as f:
            config = json.load(f)

    is_val = config.get("validation", False) or df_all["sample_id"].nunique() <= 50
    reps = config.get("repetitions", 1)
    run_id = summary.get("run_id", "unknown_run")

    plots_subdir = "validation" if is_val else "final"
    out_dir = os.path.join(target_dir, "plots", plots_subdir)
    plots_root = os.path.join(target_dir, "plots")

    print(f"Generating research-grade publication plots into: {out_dir}")
    generated = []

    generated.extend(plot_01_accuracy_by_strategy(df_all, reps, out_dir))
    generated.extend(plot_02_energy_by_strategy(df_success, reps, out_dir))
    generated.extend(plot_02b_prefill_vs_decode_stacked(df_success, reps, out_dir))
    generated.extend(plot_03_latency_by_strategy(df_success, reps, out_dir))
    generated.extend(plot_04_token_composition_by_strategy(df_success, reps, out_dir))
    generated.extend(plot_05_accuracy_energy_pareto(df_all, reps, out_dir))
    generated.extend(plot_06_accuracy_latency(df_all, reps, out_dir))
    generated.extend(plot_07_prefill_energy_vs_input_tokens(df_success, reps, out_dir))
    generated.extend(plot_08_decode_energy_vs_output_tokens(df_success, reps, out_dir))
    generated.extend(plot_09_token_energy_efficiency(df_success, reps, out_dir))
    generated.extend(plot_10_efficiency_by_strategy(df_all, reps, out_dir))
    generated.extend(plot_11_energy_distribution(df_success, reps, out_dir))
    generated.extend(plot_12_latency_distribution(df_success, reps, out_dir))
    generated.extend(plot_13_truncation_rate(df_all, reps, out_dir))

    if is_val:
        diag = plot_validation_overview(df_all, summary, out_dir)
        generated.append(diag)

    # Save metadata in plots root directory
    save_plot_metadata(plots_root, run_id, results_file, df_all, reps)

    print(f"Successfully generated {len(generated)} plot artifact(s).")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate research-grade publication plots for PromptEnergy-Bench.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to specific run directory")
    parser.add_argument("--results-jsonl", type=str, default=None, help="Direct path to results.jsonl")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--device", type=str, default=None, help="Filter by device name")
    parser.add_argument("--run-id", type=str, default=None, help="Filter by run ID")
    args = parser.parse_args()

    results_file = args.results_jsonl
    target_dir = args.run_dir

    if not results_file and target_dir:
        results_file = os.path.join(target_dir, "results.jsonl")

    if not results_file:
        candidates = glob.glob("results/**/results.jsonl", recursive=True)
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            results_file = candidates[0]
            target_dir = os.path.dirname(results_file)
            print(f"Auto-detected latest experiment: {results_file}")
        else:
            print("No results.jsonl found.")
            return

    if not target_dir:
        target_dir = os.path.dirname(results_file)

    generate_all_plots_for_run(
        target_dir=target_dir,
        results_file=results_file,
        model_filter=args.model,
        device_filter=args.device,
        run_id_filter=args.run_id
    )


if __name__ == "__main__":
    main()