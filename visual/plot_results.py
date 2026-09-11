"""Publication-Grade Visualization Suite for PromptEnergy-Bench.

Generates 13 publication figures (PNG @ 300 DPI, vector PDF, and SVG)
plus a validation diagnostic dashboard:
  01_accuracy_by_strategy.*
  02_energy_by_strategy.*
  03_latency_by_strategy.*
  04_token_composition_by_strategy.*
  05_accuracy_energy_pareto.*
  06_accuracy_latency.*
  07_energy_vs_output_tokens.*
  08_thinking_tokens_vs_energy.*
  09_thinking_tokens_vs_accuracy.*
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
    "legend.framealpha": 0.9,
    "figure.autolayout": False
})

STRATEGY_ORDER = [
    "zero_shot_direct",
    "few_shot_3",
    "zero_shot_cot",
    "short_cot",
    "long_cot"
]

STRATEGY_LABELS = {
    "zero_shot_direct": "Zero-shot Direct",
    "few_shot_3": "Few-shot (3)",
    "zero_shot_cot": "Zero-shot CoT",
    "short_cot": "Short CoT",
    "long_cot": "Long CoT"
}

# Consistent color mapping
_palette_colors = sns.color_palette("colorblind", n_colors=len(STRATEGY_ORDER))
STRATEGY_PALETTE = dict(zip(STRATEGY_ORDER, _palette_colors))
STRATEGY_DISPLAY_PALETTE = {STRATEGY_LABELS[k]: v for k, v in STRATEGY_PALETTE.items()}


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
                records.append(json.loads(line))

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

    # Map strategy display names and categories
    if "strategy" in df_all.columns:
        df_all["strategy_display"] = df_all["strategy"].map(lambda s: STRATEGY_LABELS.get(s, s))
        # Ensure categorical order
        cat_order = [STRATEGY_LABELS.get(s, s) for s in STRATEGY_ORDER if s in df_all["strategy"].unique()]
        df_all["strategy_display"] = pd.Categorical(df_all["strategy_display"], categories=cat_order, ordered=True)

    df_success = df_all[df_all["status"] == "success"].copy() if "status" in df_all.columns else df_all.copy()
    return df_all, df_success


# ==============================================================================
# 13 RESEARCH-PAPER FIGURES
# ==============================================================================

def plot_01_accuracy_by_strategy(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 1: Accuracy (%) by Prompt Strategy."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    acc_df = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True)["answer_correct"].mean()
    acc_df["accuracy_pct"] = acc_df["answer_correct"] * 100.0

    bars = sns.barplot(
        data=acc_df,
        x="strategy_display",
        y="accuracy_pct",
        hue="strategy_display",
        palette=STRATEGY_DISPLAY_PALETTE,
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
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_ylim(0, max(100, (acc_df["accuracy_pct"].max() + 15) if not acc_df.empty else 100))
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "01_accuracy_by_strategy")


def plot_02_energy_by_strategy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 2: Mean Energy per Inference (J) by Strategy."""
    if "energy_total_j" not in df_success.columns or df_success["energy_total_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # If reps > 1 show 95% CI, if reps == 1 show mean without fake CI
    errorbar = ("ci", 95) if reps > 1 else None
    bars = sns.barplot(
        data=df_success,
        x="strategy_display",
        y="energy_total_j",
        hue="strategy_display",
        palette=STRATEGY_DISPLAY_PALETTE,
        legend=False,
        errorbar=errorbar,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    # Annotate mean energy
    for p in bars.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f} J", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

    ax.set_ylabel("Mean Energy per Inference (J)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "02_energy_by_strategy")


def plot_03_latency_by_strategy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 3: Latency Decomposition (TTFT vs Generation Latency vs Total Latency)."""
    needed = ["strategy", "strategy_display", "total_latency_ms"]
    if not all(c in df_success.columns for c in needed) or df_success["total_latency_ms"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # Aggregate latencies per strategy
    agg = df_success.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "ttft_ms": "mean",
        "generation_latency_ms": "mean",
        "total_latency_ms": "mean"
    })

    # Reshape for grouped bar plot
    melted = pd.melt(
        agg,
        id_vars=["strategy_display"],
        value_vars=["ttft_ms", "generation_latency_ms", "total_latency_ms"],
        var_name="latency_phase",
        value_name="latency_ms"
    )
    phase_labels = {
        "ttft_ms": "Time to First Token (TTFT)",
        "generation_latency_ms": "Generation Latency",
        "total_latency_ms": "Total Latency"
    }
    melted["phase_display"] = melted["latency_phase"].map(phase_labels)

    sns.barplot(
        data=melted,
        x="strategy_display",
        y="latency_ms",
        hue="phase_display",
        palette="Blues_d",
        edgecolor="#333333",
        linewidth=0.8,
        ax=ax
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Prompting Strategy")
    ax.legend(title="Latency Component", loc="upper left")
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
    width = 0.55

    th_vals = agg["thinking_tokens"].values
    vis_vals = agg["visible_output_tokens"].values

    p1 = ax.bar(x_indices, th_vals, width, label="Thinking Tokens", color="#9467BD", edgecolor="#333333", linewidth=0.8)
    p2 = ax.bar(x_indices, vis_vals, width, bottom=th_vals, label="Visible Output Tokens", color="#1F77B4", edgecolor="#333333", linewidth=0.8)

    # Annotate total tokens above each bar
    for i, (tot, th, vis) in enumerate(zip(agg["output_tokens"], th_vals, vis_vals)):
        height = th + vis
        ax.annotate(f"{tot:.0f}", (i, height), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(agg["strategy_display"], rotation=15, ha="right")
    ax.set_ylabel("Mean Token Count")
    ax.set_xlabel("Prompting Strategy")
    ax.legend(loc="upper left")
    return save_publication_figure(fig, out_dir, "04_token_composition_by_strategy")


def plot_05_accuracy_energy_pareto(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 5: Accuracy–Energy Pareto Frontier."""
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

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    palette = [STRATEGY_PALETTE.get(s, "#4C72B0") for s in agg["strategy"]]

    # Scatter of all strategies
    sns.scatterplot(
        data=agg,
        x="energy_total_j",
        y="accuracy_pct",
        hue="strategy",
        palette=STRATEGY_PALETTE,
        s=160,
        legend=False,
        edgecolor="#333333",
        linewidth=1.2,
        ax=ax
    )

    # Draw dashed Pareto line
    if len(pareto_pts) > 1:
        p_df = pd.DataFrame(pareto_pts).sort_values("energy_j")
        ax.plot(p_df["energy_j"], p_df["accuracy"], linestyle="--", color="#D62728", linewidth=1.8, label="Pareto Frontier")

    # Annotate strategies
    for _, row in agg.iterrows():
        strat = row["strategy"]
        is_p = strat in pareto_strats
        label = f"{STRATEGY_LABELS.get(strat, strat)}{' (Pareto)' if is_p else ''}"
        ax.annotate(
            label,
            (row["energy_total_j"], row["accuracy_pct"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold" if is_p else "normal"
        )

    ax.set_xlabel("Mean Energy per Inference (J)")
    ax.set_ylabel("Accuracy (%)")
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

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.scatterplot(
        data=agg,
        x="total_latency_ms",
        y="accuracy_pct",
        hue="strategy",
        palette=STRATEGY_PALETTE,
        s=150,
        legend=False,
        edgecolor="#333333",
        linewidth=1.2,
        ax=ax
    )

    for _, row in agg.iterrows():
        ax.annotate(
            STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
            (row["total_latency_ms"], row["accuracy_pct"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9.5
        )

    ax.set_xlabel("Mean Total Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    return save_publication_figure(fig, out_dir, "06_accuracy_latency")


def plot_07_energy_vs_output_tokens(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 7: Energy vs Total Output Tokens."""
    if "energy_total_j" not in df_success.columns or "output_tokens" not in df_success.columns:
        return []

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sns.scatterplot(
        data=df_success,
        x="output_tokens",
        y="energy_total_j",
        hue="strategy",
        palette=STRATEGY_PALETTE,
        alpha=0.75,
        s=45,
        ax=ax
    )

    ax.set_xlabel("Total Output Tokens (Inference-Level)")
    ax.set_ylabel("Estimated System Energy (J)")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_publication_figure(fig, out_dir, "07_energy_vs_output_tokens")


def plot_08_thinking_tokens_vs_energy(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 8: Observed Thinking Tokens vs Energy."""
    th_df = df_success[df_success["thinking_tokens"].notna() & (df_success["thinking_tokens"] > 0)].copy()
    if th_df.empty or "energy_total_j" not in th_df.columns:
        return []

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sns.scatterplot(
        data=th_df,
        x="thinking_tokens",
        y="energy_total_j",
        hue="strategy",
        palette=STRATEGY_PALETTE,
        alpha=0.8,
        s=50,
        ax=ax
    )

    # Optional regression line if statistically informative
    if len(th_df) > 10:
        sns.regplot(
            data=th_df,
            x="thinking_tokens",
            y="energy_total_j",
            scatter=False,
            color="#555555",
            line_kws={"linestyle": "--", "linewidth": 1.5},
            ax=ax
        )

    ax.set_xlabel("Observed Thinking Tokens")
    ax.set_ylabel("Estimated System Energy (J)")
    ax.set_title("Association between observed thinking-token count and estimated system energy", fontsize=10, style="italic")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_publication_figure(fig, out_dir, "08_thinking_tokens_vs_energy")


def plot_09_thinking_tokens_vs_accuracy(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 9: Thinking Tokens vs Accuracy / Correctness."""
    agg = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "thinking_tokens": lambda x: x.dropna().mean() if not x.dropna().empty else 0.0,
        "answer_correct": "mean"
    })
    agg["accuracy_pct"] = agg["answer_correct"] * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.scatterplot(
        data=agg,
        x="thinking_tokens",
        y="accuracy_pct",
        hue="strategy",
        palette=STRATEGY_PALETTE,
        s=160,
        legend=False,
        edgecolor="#333333",
        linewidth=1.2,
        ax=ax
    )

    for _, row in agg.iterrows():
        ax.annotate(
            STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
            (row["thinking_tokens"], row["accuracy_pct"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9.5
        )

    ax.set_xlabel("Mean Thinking Tokens")
    ax.set_ylabel("Accuracy (%)")
    return save_publication_figure(fig, out_dir, "09_thinking_tokens_vs_accuracy")


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

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = sns.barplot(
        data=agg,
        x="strategy_display",
        y="accuracy_per_joule",
        hue="strategy_display",
        palette=STRATEGY_DISPLAY_PALETTE,
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
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "10_efficiency_by_strategy")


def plot_11_energy_distribution(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 11: Distribution of Energy (Stripplot for n=1, Boxplot for repeated)."""
    if "energy_total_j" not in df_success.columns or df_success["energy_total_j"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = [STRATEGY_PALETTE.get(s, "#4C72B0") for s in STRATEGY_ORDER if s in df_success["strategy"].unique()]

    if reps > 1:
        sns.boxplot(data=df_success, x="strategy_display", y="energy_total_j", hue="strategy_display", palette=palette, legend=False, ax=ax, width=0.4, fliersize=2)
        sns.stripplot(data=df_success, x="strategy_display", y="energy_total_j", color="#222222", alpha=0.3, size=3, jitter=0.2, ax=ax)
    else:
        sns.stripplot(data=df_success, x="strategy_display", y="energy_total_j", hue="strategy_display", palette=palette, legend=False, size=5, jitter=0.25, alpha=0.7, ax=ax)

    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "11_energy_distribution")


def plot_12_latency_distribution(df_success: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 12: Distribution of Latency."""
    if "total_latency_ms" not in df_success.columns or df_success["total_latency_ms"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = [STRATEGY_PALETTE.get(s, "#4C72B0") for s in STRATEGY_ORDER if s in df_success["strategy"].unique()]

    if reps > 1:
        sns.boxplot(data=df_success, x="strategy_display", y="total_latency_ms", hue="strategy_display", palette=palette, legend=False, ax=ax, width=0.4, fliersize=2)
        sns.stripplot(data=df_success, x="strategy_display", y="total_latency_ms", color="#222222", alpha=0.3, size=3, jitter=0.2, ax=ax)
    else:
        sns.stripplot(data=df_success, x="strategy_display", y="total_latency_ms", hue="strategy_display", palette=palette, legend=False, size=5, jitter=0.25, alpha=0.7, ax=ax)

    ax.set_ylabel("Total Latency (ms)")
    ax.set_xlabel("Prompting Strategy")
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "12_latency_distribution")


def plot_13_truncation_rate(df_all: pd.DataFrame, reps: int, out_dir: str) -> List[str]:
    """Figure 13: Generation Truncation Rate (%) by Strategy."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    trunc_df = df_all.groupby(["strategy", "strategy_display"], as_index=False, observed=True).agg({
        "generation_truncated": lambda x: (sum(1 for v in x if v is True) / len(x)) * 100.0 if len(x) > 0 else 0.0
    })

    bars = sns.barplot(
        data=trunc_df,
        x="strategy_display",
        y="generation_truncated",
        hue="strategy_display",
        palette=STRATEGY_DISPLAY_PALETTE,
        legend=False,
        edgecolor="#333333",
        linewidth=1.0,
        ax=ax
    )

    for p in bars.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., max(0.5, height)),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

    ax.set_ylabel("Truncation Rate (%)")
    ax.set_xlabel("Prompting Strategy")
    ax.set_ylim(0, max(10, trunc_df["generation_truncated"].max() + 10))
    plt.xticks(rotation=15, ha="right")
    return save_publication_figure(fig, out_dir, "13_truncation_rate")


def plot_validation_overview(df_all: pd.DataFrame, summary: Dict[str, Any], out_dir: str) -> str:
    """Creates a concise, publication-style multi-panel validation diagnostic dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    fig.suptitle("Pipeline Validation Diagnostic Overview", fontsize=14, fontweight="bold", y=0.98)

    # 1. Accuracy Panel
    acc_df = df_all.groupby("strategy_display", observed=True)["answer_correct"].mean().reset_index()
    acc_df["acc_pct"] = acc_df["answer_correct"] * 100.0
    sns.barplot(data=acc_df, x="strategy_display", y="acc_pct", hue="strategy_display", legend=False, ax=axes[0, 0], palette="crest")
    axes[0, 0].set_title("Accuracy (%)", fontsize=11)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=25)

    # 2. Truncation Rate Panel
    trunc_df = df_all.groupby("strategy_display", observed=True)["generation_truncated"].apply(
        lambda s: (sum(1 for v in s if v is True) / len(s)) * 100.0 if len(s) > 0 else 0.0
    ).reset_index()
    sns.barplot(data=trunc_df, x="strategy_display", y="generation_truncated", hue="strategy_display", legend=False, ax=axes[0, 1], palette="Reds_r")
    axes[0, 1].set_title("Truncation Rate (%)", fontsize=11)
    axes[0, 1].set_ylabel("Truncated (%)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=25)

    # 3. Energy Consumption Panel
    if "energy_total_j" in df_all.columns:
        e_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=e_df, x="strategy_display", y="energy_total_j", hue="strategy_display", legend=False, ax=axes[0, 2], palette="viridis")
        axes[0, 2].set_title("Mean Energy (J)", fontsize=11)
        axes[0, 2].set_ylabel("Energy (Joules)")
        axes[0, 2].set_xlabel("")
        axes[0, 2].tick_params(axis="x", rotation=25)

    # 4. Latency Panel
    if "total_latency_ms" in df_all.columns:
        l_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=l_df, x="strategy_display", y="total_latency_ms", hue="strategy_display", legend=False, ax=axes[1, 0], palette="mako")
        axes[1, 0].set_title("Mean Latency (ms)", fontsize=11)
        axes[1, 0].set_ylabel("Total Latency (ms)")
        axes[1, 0].set_xlabel("")
        axes[1, 0].tick_params(axis="x", rotation=25)

    # 5. Output Tokens Panel
    if "output_tokens" in df_all.columns:
        tok_df = df_all[df_all["status"] == "success"]
        sns.barplot(data=tok_df, x="strategy_display", y="output_tokens", hue="strategy_display", legend=False, ax=axes[1, 1], palette="flare")
        axes[1, 1].set_title("Mean Output Tokens", fontsize=11)
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
        f" • Truncated Generations: {metrics.get('truncated_generations', 0)}\n"
        f" • Truncation Rate: {metrics.get('truncation_rate', 0.0) * 100.0:.2f}%\n"
        f" • Mean Energy: {metrics.get('mean_energy_j', 'N/A')} J\n"
        f" • Mean Latency: {metrics.get('mean_latency_ms', 'N/A')} ms"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.95, diag_text, transform=axes[1, 2].transAxes,
                    fontsize=10.5, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#CCCCCC"))

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
    """Generates all 13 research figures + diagnostic overview and metadata for a run."""
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
    generated.extend(plot_03_latency_by_strategy(df_success, reps, out_dir))
    generated.extend(plot_04_token_composition_by_strategy(df_success, reps, out_dir))
    generated.extend(plot_05_accuracy_energy_pareto(df_all, reps, out_dir))
    generated.extend(plot_06_accuracy_latency(df_all, reps, out_dir))
    generated.extend(plot_07_energy_vs_output_tokens(df_success, reps, out_dir))
    generated.extend(plot_08_thinking_tokens_vs_energy(df_success, reps, out_dir))
    generated.extend(plot_09_thinking_tokens_vs_accuracy(df_all, reps, out_dir))
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