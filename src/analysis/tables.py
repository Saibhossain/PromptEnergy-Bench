"""Publication-Grade Table Generation for PromptEnergy-Bench.

Generates Tables 1-5 in CSV, Markdown (.md), and LaTeX (.tex) formats:
- Table 1: Experimental Configuration (Reproducibility)
- Table 2: Prompt Strategy Comparison (Core Benchmark)
- Table 3: Energy-Accuracy-Latency Trade-off (Pareto Decision Matrix)
- Table 4: Marginal Energy Gain (MEG Analysis)
- Table 5: Statistical Summary (Distributions & Confidence Intervals)
"""

import os
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.analysis.pareto import compute_pareto_frontier


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


def _get_strategies(strat_summaries: Dict[str, Any]) -> List[str]:
    """Returns sorted strategies adhering to standard order first, followed by custom strategies."""
    strats = [s for s in STRATEGY_ORDER if s in strat_summaries]
    for s in strat_summaries:
        if s not in strats:
            strats.append(s)
    return strats


def get_strategy_display_name(strat: str) -> str:
    """Returns human-readable strategy display name."""
    if strat in STRATEGY_DISPLAY:
        return STRATEGY_DISPLAY[strat]
    if strat.startswith("ctx_"):
        return f"Context {strat[4:]} tokens"
    if strat.startswith("rag_top_"):
        return f"BM25 RAG (top-{strat[8:]})"
    return strat.replace("_", " ").title()


def _format_val(val: Any, decimals: int = 2, fallback: str = "N/A") -> str:
    """Safely formats float, int, or None values without coercing None/NaN to 0."""
    if val is None:
        return fallback
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return fallback
        return f"{val:.{decimals}f}"
    return str(val)


def _save_table_formats(df: pd.DataFrame, base_dir: str, file_stem: str, caption: str = "") -> Dict[str, str]:
    """Saves a dataframe in CSV, Markdown, and LaTeX formats."""
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, f"{file_stem}.csv")
    md_path = os.path.join(base_dir, f"{file_stem}.md")
    tex_path = os.path.join(base_dir, f"{file_stem}.tex")

    # CSV
    df.to_csv(csv_path, index=False)

    # Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {caption}\n\n" if caption else "")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            cols = [str(c) for c in df.columns]
            lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
            for _, r in df.iterrows():
                lines.append("| " + " | ".join(str(v).replace("\n", " ") for v in r) + " |")
            f.write("\n".join(lines))
        f.write("\n")

    # LaTeX
    with open(tex_path, "w", encoding="utf-8") as f:
        try:
            f.write(df.to_latex(index=False, caption=caption, label=f"tab:{file_stem}", escape=True))
        except Exception:
            cols = [str(c) for c in df.columns]
            col_spec = "l" * len(cols)
            lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}" if caption else "",
                f"\\label{{tab:{file_stem}}}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\hline",
                " & ".join(cols) + " \\\\",
                "\\hline"
            ]
            for _, r in df.iterrows():
                row_str = " & ".join(str(v).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_") for v in r) + " \\\\"
                lines.append(row_str)
            lines.extend([
                "\\hline",
                "\\end{tabular}",
                "\\end{table}\n"
            ])
            f.write("\n".join(lines))

    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def generate_table_1_config(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    tables_dir: str
) -> Dict[str, str]:
    """Table 1: Experimental Configuration (Reproducibility)."""
    device = metadata.get("device", {})
    backend = metadata.get("backend", {})
    meas = metadata.get("measurement", {})
    sampling = config.get("sampling", {})
    dataset = config.get("dataset", {})

    gen_cfg = config.get("generation", {})
    strat_max = gen_cfg.get("strategy_max_tokens", {})
    if strat_max:
        max_tok_str = " / ".join(f"{get_strategy_display_name(k)}: {v}" for k, v in strat_max.items())
    else:
        max_tok_str = str(sampling.get("max_tokens", 512))

    rows = [
        {"Parameter": "Experiment", "Value": config.get("experiment_name", metadata.get("experiment_name", "primary_exp_gsm8k"))},
        {"Parameter": "Dataset", "Value": dataset.get("name", "gsm8k").upper()},
        {"Parameter": "Evaluation Split", "Value": dataset.get("evaluation_split", "test").upper()},
        {"Parameter": "Model", "Value": config.get("model", {}).get("name", backend.get("model_name", "N/A"))},
        {"Parameter": "Backend Operator", "Value": config.get("model", {}).get("backend", backend.get("operator", "N/A"))},
        {"Parameter": "Model Format", "Value": config.get("model", {}).get("format", backend.get("model_format", "N/A"))},
        {"Parameter": "Device", "Value": device.get("name", "N/A")},
        {"Parameter": "Operating System", "Value": device.get("os", "N/A")},
        {"Parameter": "CPU", "Value": device.get("cpu", "N/A")},
        {"Parameter": "GPU", "Value": f"{device.get('gpu_name', 'N/A')} (Count: {device.get('gpu_count', 1)})" if device.get("gpu_available") else "None"},
        {"Parameter": "System RAM", "Value": f"{device.get('ram_gb', 'N/A')} GB"},
        {"Parameter": "Evaluation Examples", "Value": str(dataset.get("evaluation_size", 50))},
        {"Parameter": "Prompt Strategies", "Value": str(len(config.get("strategies", [5])))},
        {"Parameter": "Warmup Runs", "Value": str(config.get("warmups", 1))},
        {"Parameter": "Repetitions", "Value": str(config.get("repetitions", 1))},
        {"Parameter": "Max Generation Tokens", "Value": max_tok_str},
        {"Parameter": "Sampling Temperature", "Value": str(sampling.get("temperature", 0.0))},
        {"Parameter": "Sampling Seed", "Value": str(sampling.get("seed", 42))},
        {"Parameter": "Energy Measurement Method", "Value": meas.get("energy_method", "apple_estimated")},
        {"Parameter": "Energy Quality / Level", "Value": f"{meas.get('energy_quality', 'software_estimate')} ({meas.get('energy_measurement_level', 'estimated_system')})"}
    ]

    df = pd.DataFrame(rows)
    return _save_table_formats(df, tables_dir, "experimental_configuration", caption="Experimental Configuration and Reproducibility Parameters")


def generate_table_2_comparison(
    summary: Dict[str, Any],
    tables_dir: str
) -> Dict[str, str]:
    """Table 2: Prompt Strategy Comparison."""
    strat_summaries = summary.get("metrics", {}).get("strategy_summaries", {})
    if not strat_summaries:
        strat_summaries = summary.get("strategy_metrics", {})

    strats = _get_strategies(strat_summaries)
    ds_name = str(summary.get("dataset", summary.get("dataset_name", "dataset"))).upper()

    rows = []
    for strat in strats:
        s_data = strat_summaries.get(strat, {})
        if not s_data:
            continue

        tot_acc = s_data.get("total_accuracy", s_data.get("accuracy"))
        tot_acc_pct = f"{tot_acc * 100.0:.2f}" if tot_acc is not None else "N/A"
        
        val_acc = s_data.get("valid_accuracy")
        val_acc_pct = f"{val_acc * 100.0:.2f}" if val_acc is not None else "N/A"

        th_tok = _format_val(s_data.get("mean_thinking_tokens"), 1)
        vis_tok = _format_val(s_data.get("mean_visible_output_tokens"), 1)
        tot_tok = _format_val(s_data.get("mean_output_tokens"), 1)
        ttft = _format_val(s_data.get("mean_ttft_ms"), 1)
        gen_lat = _format_val(s_data.get("mean_generation_latency_ms"), 1)
        tot_lat = _format_val(s_data.get("mean_total_latency_ms"), 1)
        mean_cpu = _format_val(s_data.get("mean_cpu_percent"), 1)
        peak_cpu = _format_val(s_data.get("peak_cpu_percent"), 1)
        ram_gb = _format_val(s_data.get("mean_ram_used_gb"), 2)
        energy = _format_val(s_data.get("mean_energy_j"), 4)
        e_per_c = _format_val(s_data.get("energy_per_correct_answer_j"), 4)
        acc_j = _format_val(s_data.get("accuracy_per_joule"), 4)
        tps = _format_val(s_data.get("tokens_per_second"), 2)
        trunc_rate = f"{s_data.get('truncation_rate', 0.0) * 100.0:.1f}"

        rows.append({
            "Strategy": get_strategy_display_name(strat),
            "Total Accuracy (%)": tot_acc_pct,
            "Valid Accuracy (%)": val_acc_pct,
            "Valid Samples": f"{s_data.get('valid_n', 0)}/{s_data.get('n', 0)}",
            "Mean Thinking Tokens": th_tok,
            "Mean Visible Output Tokens": vis_tok,
            "Mean Total Output Tokens": tot_tok,
            "Throughput (tok/s)": tps,
            "TTFT (ms)": ttft,
            "Total Latency (ms)": tot_lat,
            "Mean CPU (%)": mean_cpu,
            "Peak CPU (%)": peak_cpu,
            "RAM (GB)": ram_gb,
            "Energy (J)": energy,
            "Energy / Correct (J)": e_per_c,
            "Accuracy / Joule": acc_j,
            "Truncation Rate (%)": trunc_rate
        })

    df = pd.DataFrame(rows)
    return _save_table_formats(df, tables_dir, "strategy_comparison", caption=f"Prompt Strategy Comparison on {ds_name}")


def generate_table_3_tradeoff(
    summary: Dict[str, Any],
    tables_dir: str
) -> Dict[str, str]:
    """Table 3: Energy-Accuracy-Latency Trade-off & Pareto Optimality."""
    strat_summaries = summary.get("metrics", {}).get("strategy_summaries", {})
    if not strat_summaries:
        strat_summaries = summary.get("strategy_metrics", {})

    strats = _get_strategies(strat_summaries)
    baseline_strat = "zero_shot_direct" if "zero_shot_direct" in strat_summaries else (strats[0] if strats else "")
    baseline = strat_summaries.get(baseline_strat, {})
    base_acc = baseline.get("total_accuracy", baseline.get("accuracy", 0.0))
    base_energy = baseline.get("mean_energy_j")
    base_lat = baseline.get("mean_total_latency_ms")

    # Determine Pareto frontier on (energy, accuracy)
    candidates = []
    for strat in strats:
        s_data = strat_summaries.get(strat)
        acc_val = s_data.get("total_accuracy", s_data.get("accuracy")) if s_data else None
        energy_val = s_data.get("mean_energy_j") if s_data else None
        if s_data and acc_val is not None and energy_val is not None:
            candidates.append({
                "strategy": strat,
                "accuracy": acc_val,
                "energy_j": energy_val,
                "latency_ms": s_data.get("mean_total_latency_ms", 0.0) or 0.0
            })

    pareto_frontier = compute_pareto_frontier(candidates) if candidates else []
    pareto_strats = {p["strategy"] for p in pareto_frontier}

    rows = []
    for strat in strats:
        s_data = strat_summaries.get(strat, {})
        if not s_data:
            continue

        acc = s_data.get("total_accuracy", s_data.get("accuracy"))
        acc_pct = f"{acc * 100.0:.2f}%" if acc is not None else "N/A"
        energy = _format_val(s_data.get("mean_energy_j"), 4)
        latency = _format_val(s_data.get("mean_total_latency_ms"), 1)
        th_tok = _format_val(s_data.get("mean_thinking_tokens"), 1)
        is_pareto = "Yes" if strat in pareto_strats else "No"

        if strat == baseline_strat:
            d_acc = "Baseline"
            d_energy = "Baseline"
            d_lat = "Baseline"
        else:
            if acc is not None and base_acc is not None:
                delta_acc_val = acc - base_acc
                d_acc = f"{delta_acc_val * 100.0:+.2f} pp"
            else:
                d_acc = "N/A"

            s_energy = s_data.get("mean_energy_j")
            if s_energy is not None and base_energy is not None:
                d_energy = f"{s_energy - base_energy:+.4f} J"
            else:
                d_energy = "N/A"

            s_lat = s_data.get("mean_total_latency_ms")
            if s_lat is not None and base_lat is not None:
                d_lat = f"{s_lat - base_lat:+.1f} ms"
            else:
                d_lat = "N/A"

        rows.append({
            "Strategy": get_strategy_display_name(strat),
            "Total Accuracy": acc_pct,
            "Energy (J)": energy,
            "Latency (ms)": latency,
            "Thinking Tokens": th_tok,
            "Pareto Optimal": is_pareto,
            "Accuracy Gain vs Baseline": d_acc,
            "Energy Increase vs Baseline": d_energy,
            "Latency Increase vs Baseline": d_lat
        })

    df = pd.DataFrame(rows)
    return _save_table_formats(df, tables_dir, "tradeoff_comparison", caption="Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis")


def generate_table_4_meg(
    summary: Dict[str, Any],
    tables_dir: str
) -> Dict[str, str]:
    """Table 4: Marginal Energy Gain (MEG).
    
    MEG(B1, B2) = [Accuracy(B2) - Accuracy(B1)] / [Energy(B2) - Energy(B1)]
    Units: Accuracy percentage points per Joule (% / J).
    """
    strat_summaries = summary.get("metrics", {}).get("strategy_summaries", {})
    if not strat_summaries:
        strat_summaries = summary.get("strategy_metrics", {})

    strats = _get_strategies(strat_summaries)
    baseline_strat = "zero_shot_direct" if "zero_shot_direct" in strat_summaries else (strats[0] if strats else "")
    base = strat_summaries.get(baseline_strat, {})
    base_acc = base.get("total_accuracy", base.get("accuracy"))
    base_energy = base.get("mean_energy_j")
    base_lat = base.get("mean_total_latency_ms")

    rows = []
    comparisons = [s for s in strats if s != baseline_strat]

    for comp in comparisons:
        c_data = strat_summaries.get(comp, {})
        if not c_data:
            continue

        c_acc = c_data.get("total_accuracy", c_data.get("accuracy"))
        c_energy = c_data.get("mean_energy_j")
        c_lat = c_data.get("mean_total_latency_ms")

        if c_acc is not None and base_acc is not None:
            d_acc_pct = (c_acc - base_acc) * 100.0
            d_acc_str = f"{d_acc_pct:+.2f} pp"
        else:
            d_acc_pct = None
            d_acc_str = "N/A"

        if c_energy is not None and base_energy is not None:
            d_energy = c_energy - base_energy
            d_energy_str = f"{d_energy:+.4f} J"
        else:
            d_energy = None
            d_energy_str = "N/A"

        if d_acc_pct is not None and d_energy is not None and abs(d_energy) > 1e-6:
            meg_val = d_acc_pct / d_energy
            meg_str = f"{meg_val:.4f} %/J"
        else:
            meg_str = "N/A"

        if c_lat is not None and base_lat is not None:
            d_lat_str = f"{c_lat - base_lat:+.1f} ms"
        else:
            d_lat_str = "N/A"

        rows.append({
            "Baseline Strategy": get_strategy_display_name(baseline_strat),
            "Comparison Strategy": get_strategy_display_name(comp),
            "Accuracy Change": d_acc_str,
            "Energy Change": d_energy_str,
            "Marginal Energy Gain (% / J)": meg_str,
            "Latency Change": d_lat_str
        })

    df = pd.DataFrame(rows)
    return _save_table_formats(df, tables_dir, "marginal_energy_gain", caption="Marginal Energy Gain (MEG) Relative to Baseline")


def generate_table_5_statistical_summary(
    records: List[Dict[str, Any]],
    repetitions: int,
    tables_dir: str
) -> Dict[str, str]:
    """Table 5: Statistical Summary (Distributions & Confidence Intervals)."""
    rows = []
    successful = [r for r in records if r.get("status") == "success"]

    strategies_in_recs = sorted(list(set(r.get("strategy") for r in records if r.get("strategy"))))
    strats = [s for s in STRATEGY_ORDER if s in strategies_in_recs]
    for s in strategies_in_recs:
        if s not in strats:
            strats.append(s)

    for strat in strats:
        strat_recs = [r for r in successful if r.get("strategy") == strat]
        if not strat_recs:
            continue

        strat_name = get_strategy_display_name(strat)
        
        # Valid numbers extraction helper
        def _valid_list(key: str) -> List[float]:
            res = []
            for r in strat_recs:
                v = r.get(key)
                if v is not None and isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                    res.append(float(v))
            return res

        metrics_map = {
            "Total Accuracy": [1.0 if r.get("answer_correct") else 0.0 for r in strat_recs],
            "Energy (J)": _valid_list("energy_total_j"),
            "Total Latency (ms)": _valid_list("total_latency_ms"),
            "Generation Latency (ms)": _valid_list("generation_latency_ms"),
            "TTFT (ms)": _valid_list("ttft_ms"),
            "Thinking Tokens": _valid_list("thinking_tokens"),
            "Output Tokens": _valid_list("output_tokens")
        }

        for metric_name, vals in metrics_map.items():
            if not vals:
                rows.append({
                    "Strategy": strat_name,
                    "Metric": metric_name,
                    "Mean": "N/A",
                    "Median": "N/A",
                    "Standard Deviation": "N/A",
                    "95% Confidence Interval": "N/A"
                })
                continue

            mean_v = statistics.mean(vals)
            med_v = statistics.median(vals)
            sd_v = statistics.stdev(vals) if len(vals) > 1 else 0.0

            if repetitions > 1 and len(vals) > 1:
                # 95% Student-t CI
                try:
                    from scipy import stats
                    sem = stats.sem(vals)
                    margin = sem * stats.t.ppf(0.975, len(vals) - 1)
                    ci_str = f"[{mean_v - margin:.2f}, {mean_v + margin:.2f}]"
                except Exception:
                    ci_str = "N/A"
            else:
                ci_str = "N/A (n=1)"

            rows.append({
                "Strategy": strat_name,
                "Metric": metric_name,
                "Mean": f"{mean_v:.2f}",
                "Median": f"{med_v:.2f}",
                "Standard Deviation": f"{sd_v:.2f}",
                "95% Confidence Interval": ci_str
            })

    df = pd.DataFrame(rows)
    return _save_table_formats(df, tables_dir, "statistical_summary", caption="Statistical Summary Across Prompting Strategies")


def generate_all_tables(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    summary: Dict[str, Any],
    tables_dir: str
) -> Dict[str, Dict[str, str]]:
    """Generates all 5 publication tables in CSV, Markdown, and LaTeX formats."""
    os.makedirs(tables_dir, exist_ok=True)
    reps = config.get("repetitions", 1)

    t1 = generate_table_1_config(metadata, config, tables_dir)
    t2 = generate_table_2_comparison(summary, tables_dir)
    t3 = generate_table_3_tradeoff(summary, tables_dir)
    t4 = generate_table_4_meg(summary, tables_dir)
    t5 = generate_table_5_statistical_summary(records, reps, tables_dir)

    return {
        "table_1_config": t1,
        "table_2_comparison": t2,
        "table_3_tradeoff": t3,
        "table_4_meg": t4,
        "table_5_statistical_summary": t5,
        "table_5_statistics": t5
    }
