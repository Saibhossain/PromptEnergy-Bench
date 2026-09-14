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


def generate_research_summary_tables(results_dir: str = "results", output_dir: str = "results/analysis") -> List[str]:
    """Generates 8 unrounded, machine-readable CSV summary files in results/analysis/ (Section 11)."""
    import json
    import csv

    os.makedirs(output_dir, exist_ok=True)
    all_records = []

    for root, _, files in os.walk(results_dir):
        if "hardware_comparison" in root or "analysis" in root:
            continue
        for f in files:
            if f in ("results.jsonl", "raw_results.jsonl"):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                try:
                                    all_records.append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    pass
                except Exception:
                    pass

    generated_files = []

    # Helper to write raw CSV
    def write_csv(filename: str, headers: List[str], rows: List[List[Any]]):
        target = os.path.join(output_dir, filename)
        with open(target, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(headers)
            writer.writerows(rows)
        generated_files.append(target)

    # 1. prompt_strategy_summary.csv
    p_headers = ["dataset", "model", "hardware", "prompt_strategy", "sample_size", "accuracy", "mean_energy_j", "median_energy_j", "std_energy_j", "mean_ttft_ms", "mean_total_latency_ms", "mean_input_tokens", "mean_output_tokens", "measurement_method"]
    p_groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in all_records:
        d = r.get("dataset", "gsm8k")
        m = r.get("model") or r.get("model_name", "unknown")
        h = r.get("hardware_identifier") or r.get("device", "unknown")
        s = r.get("strategy") or r.get("prompt_strategy", "unknown")
        p_groups.setdefault((d, m, h, s), []).append(r)

    p_rows = []
    for (d, m, h, s), recs in sorted(p_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        ttft = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms")]
        lat = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
        it = [float(r["input_tokens"]) for r in recs if r.get("input_tokens") is not None]
        ot = [float(r["output_tokens"]) for r in recs if r.get("output_tokens") is not None]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        method = recs[0].get("energy_measurement_method", "software_estimated")
        p_rows.append([
            d, m, h, s, len(recs),
            np.mean(acc) if acc else None,
            np.mean(e) if e else None,
            np.median(e) if e else None,
            np.std(e) if len(e) > 1 else 0.0,
            np.mean(ttft) if ttft else None,
            np.mean(lat) if lat else None,
            np.mean(it) if it else None,
            np.mean(ot) if ot else None,
            method
        ])
    write_csv("prompt_strategy_summary.csv", p_headers, p_rows)

    # 2. model_summary.csv
    m_headers = ["dataset", "model", "hardware", "sample_size", "accuracy", "mean_energy_j", "mean_throughput_tok_s", "mean_ttft_ms", "mean_latency_ms", "measurement_method"]
    m_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in all_records:
        d = r.get("dataset", "gsm8k")
        m = r.get("model") or r.get("model_name", "unknown")
        h = r.get("hardware_identifier") or r.get("device", "unknown")
        m_groups.setdefault((d, m, h), []).append(r)

    m_rows = []
    for (d, m, h), recs in sorted(m_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        ttft = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms")]
        lat = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
        ot = [float(r["output_tokens"]) for r in recs if r.get("output_tokens") is not None]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        # Throughput
        tp_list = [(ot[i] / (lat[i]/1000.0)) for i in range(min(len(ot), len(lat))) if lat[i] > 0]
        method = recs[0].get("energy_measurement_method", "software_estimated")
        m_rows.append([
            d, m, h, len(recs),
            np.mean(acc) if acc else None,
            np.mean(e) if e else None,
            np.mean(tp_list) if tp_list else None,
            np.mean(ttft) if ttft else None,
            np.mean(lat) if lat else None,
            method
        ])
    write_csv("model_summary.csv", m_headers, m_rows)

    # 3. context_scaling_summary.csv
    ctx_headers = ["dataset", "model", "hardware", "target_context_tokens", "actual_context_tokens", "sample_size", "accuracy", "mean_energy_j", "mean_ttft_ms", "mean_latency_ms"]
    ctx_recs = [r for r in all_records if str(r.get("strategy", "")).startswith("ctx_") or r.get("context_target_tokens") is not None]
    ctx_groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for r in ctx_recs:
        d = r.get("dataset", "gsm8k")
        m = r.get("model") or r.get("model_name", "unknown")
        h = r.get("hardware_identifier") or r.get("device", "unknown")
        target = r.get("context_target_tokens")
        if target is None:
            strat = r.get("strategy", "")
            if strat.startswith("ctx_"):
                try:
                    target = int(strat[4:])
                except ValueError:
                    target = 0
            else:
                target = 0
        ctx_groups.setdefault((d, m, h, int(target)), []).append(r)

    ctx_rows = []
    for (d, m, h, target), recs in sorted(ctx_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        ttft = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms")]
        lat = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
        act_ctx = [float(r["actual_context_tokens"]) for r in recs if r.get("actual_context_tokens") is not None]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        ctx_rows.append([
            d, m, h, target,
            np.mean(act_ctx) if act_ctx else target,
            len(recs),
            np.mean(acc) if acc else None,
            np.mean(e) if e else None,
            np.mean(ttft) if ttft else None,
            np.mean(lat) if lat else None
        ])
    write_csv("context_scaling_summary.csv", ctx_headers, ctx_rows)

    # 4. rag_summary.csv
    rag_headers = ["dataset", "model", "hardware", "top_k", "sample_size", "accuracy", "mean_retrieval_latency_ms", "mean_generation_latency_ms", "mean_total_latency_ms", "mean_energy_j"]
    rag_recs = [r for r in all_records if str(r.get("strategy", "")).startswith("rag_top_") or r.get("top_k") is not None]
    rag_groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for r in rag_recs:
        d = r.get("dataset", "gsm8k")
        m = r.get("model") or r.get("model_name", "unknown")
        h = r.get("hardware_identifier") or r.get("device", "unknown")
        k = r.get("top_k", 1)
        rag_groups.setdefault((d, m, h, int(k)), []).append(r)

    rag_rows = []
    for (d, m, h, k), recs in sorted(rag_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        ret_lat = [float(r["retrieval_latency_ms"]) for r in recs if r.get("retrieval_latency_ms") is not None]
        gen_lat = [float(r["generation_latency_ms"]) for r in recs if r.get("generation_latency_ms") is not None]
        tot_lat = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms") is not None]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        rag_rows.append([
            d, m, h, k, len(recs),
            np.mean(acc) if acc else None,
            np.mean(ret_lat) if ret_lat else None,
            np.mean(gen_lat) if gen_lat else None,
            np.mean(tot_lat) if tot_lat else None,
            np.mean(e) if e else None
        ])
    write_csv("rag_summary.csv", rag_headers, rag_rows)

    # 5. hardware_summary.csv
    hw_headers = ["hardware", "model", "prompt_strategy", "sample_size", "accuracy", "mean_energy_j", "std_energy_j", "mean_ttft_ms", "mean_latency_ms", "measurement_method"]
    hw_rows = []
    for (d, m, h, s), recs in sorted(p_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        ttft = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms")]
        lat = [float(r["total_latency_ms"]) for r in recs if r.get("total_latency_ms")]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        method = recs[0].get("energy_measurement_method", "software_estimated")
        hw_rows.append([
            h, m, s, len(recs),
            np.mean(acc) if acc else None,
            np.mean(e) if e else None,
            np.std(e) if len(e) > 1 else 0.0,
            np.mean(ttft) if ttft else None,
            np.mean(lat) if lat else None,
            method
        ])
    write_csv("hardware_summary.csv", hw_headers, hw_rows)

    # 6. hardware_prompt_comparison.csv
    write_csv("hardware_prompt_comparison.csv", p_headers, p_rows)

    # 7. energy_accuracy_summary.csv
    ea_headers = ["prompt_strategy", "model", "hardware", "accuracy", "mean_energy_j", "energy_per_correct_j", "accuracy_per_joule"]
    ea_rows = []
    for (d, m, h, s), recs in sorted(p_groups.items()):
        e = [float(r["energy_total_j"]) for r in recs if r.get("energy_total_j")]
        acc = [1.0 if r.get("answer_correct") is True else 0.0 for r in recs]
        mean_e = np.mean(e) if e else 0.0
        acc_pct = np.mean(acc) if acc else 0.0
        corr_count = sum(acc)
        tot_e = sum(e)
        epc = (tot_e / corr_count) if corr_count > 0 else None
        apj = (acc_pct / mean_e) if mean_e > 0 else 0.0
        ea_rows.append([s, m, h, acc_pct, mean_e, epc, apj])
    write_csv("energy_accuracy_summary.csv", ea_headers, ea_rows)

    # 8. pareto_frontier_summary.csv
    pareto_headers = ["prompt_strategy", "model", "hardware", "accuracy", "mean_energy_j", "mean_latency_ms", "pareto_optimal"]
    pareto_rows = []
    for (d, m, h), recs in sorted(m_groups.items()):
        # Compute Pareto for this model & hardware
        sub_strats: Dict[str, Tuple[float, float, float]] = {}
        for r in recs:
            s = r.get("strategy") or r.get("prompt_strategy", "unknown")
            e_val = r.get("energy_total_j")
            lat_val = r.get("total_latency_ms")
            acc_val = 1.0 if r.get("answer_correct") is True else 0.0
            if e_val and lat_val:
                if s not in sub_strats:
                    sub_strats[s] = ([], [], [])
                sub_strats[s][0].append(float(e_val))
                sub_strats[s][1].append(float(acc_val))
                sub_strats[s][2].append(float(lat_val))

        # Check dominance
        pts = []
        for s, (e_list, acc_list, lat_list) in sub_strats.items():
            pts.append({
                "strat": s,
                "e": np.mean(e_list),
                "acc": np.mean(acc_list),
                "lat": np.mean(lat_list)
            })

        for p in pts:
            # A point p is dominated if exists q with e_q <= e_p, lat_q <= lat_p, acc_q >= acc_p (with at least one strict)
            is_dom = False
            for q in pts:
                if q == p:
                    continue
                if q["e"] <= p["e"] and q["lat"] <= p["lat"] and q["acc"] >= p["acc"]:
                    if q["e"] < p["e"] or q["lat"] < p["lat"] or q["acc"] > p["acc"]:
                        is_dom = True
                        break
            pareto_rows.append([p["strat"], m, h, p["acc"], p["e"], p["lat"], not is_dom])

    write_csv("pareto_frontier_summary.csv", pareto_headers, pareto_rows)
    return generated_files
