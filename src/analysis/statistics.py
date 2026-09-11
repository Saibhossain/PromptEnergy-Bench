"""Statistical hypothesis testing and confidence interval estimators."""

import math
import statistics
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats


def calculate_summary_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """Calculates mean, median, standard deviation, and IQR."""
    if not values:
        return {"mean": None, "median": None, "std": None, "iqr": None}
    
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"mean": None, "median": None, "std": None, "iqr": None}

    mean_val = statistics.mean(clean)
    median_val = statistics.median(clean)
    std_val = statistics.stdev(clean) if len(clean) > 1 else 0.0
    q75, q25 = np.percentile(clean, [75, 25])
    iqr_val = float(q75 - q25)

    return {
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "iqr": round(iqr_val, 4),
        "count": len(clean)
    }


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[Optional[float], Optional[float]]:
    """Computes two-sided confidence interval using Student-t distribution."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n < 2:
        return (None, None)

    mean = statistics.mean(clean)
    sem = stats.sem(clean)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (round(mean - margin, 4), round(mean + margin, 4))


def bootstrap_ci(
    values: List[float],
    num_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[Optional[float], Optional[float]]:
    """Computes non-parametric bootstrap confidence interval for the mean."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return (None, None)

    rng = np.random.default_rng(seed)
    n = len(clean)
    boot_means = [np.mean(rng.choice(clean, size=n, replace=True)) for _ in range(num_bootstrap)]
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return (round(lower, 4), round(upper, 4))


def compute_cohens_d(sample_a: List[float], sample_b: List[float]) -> Optional[float]:
    """Computes Cohen's d effect size for two paired/matched continuous samples."""
    clean_pairs = [(a, b) for a, b in zip(sample_a, sample_b) if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)]
    if len(clean_pairs) < 2:
        return None
    a_vals = [p[0] for p in clean_pairs]
    b_vals = [p[1] for p in clean_pairs]
    diffs = [a - b for a, b in clean_pairs]
    sd_diff = statistics.stdev(diffs)
    if sd_diff < 1e-9:
        return 0.0
    return round(statistics.mean(diffs) / sd_diff, 4)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m <= 1:
        return p_values

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = round(min(1.0, cum_max), 6)

    return adjusted


def paired_wilcoxon_test(
    sample_a: List[float],
    sample_b: List[float]
) -> Dict[str, Any]:
    """Runs Wilcoxon signed-rank test for paired continuous measurements."""
    clean_pairs = [(a, b) for a, b in zip(sample_a, sample_b) if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)]
    if len(clean_pairs) < 5:
        return {"statistic": None, "p_value": None, "valid": False, "note": "Insufficient paired samples"}

    a_clean = [p[0] for p in clean_pairs]
    b_clean = [p[1] for p in clean_pairs]
    diffs = [a - b for a, b in clean_pairs]
    if all(abs(d) < 1e-9 for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0, "valid": True, "note": "Identical paired samples"}

    try:
        res = stats.wilcoxon(a_clean, b_clean)
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "significant_005": bool(res.pvalue < 0.05),
            "valid": True
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "valid": False, "error": str(e)}


def build_paired_comparison_dataframe(records: List[Dict[str, Any]]) -> Any:
    """Builds question-level aligned dataframe across strategies.
    
    Columns:
        sample_id, strategy, correct, energy, latency, ttft,
        thinking_tokens, visible_output_tokens, output_tokens, truncated
    """
    import pandas as pd
    rows = []
    for r in records:
        rows.append({
            "sample_id": r.get("sample_id"),
            "strategy": r.get("strategy"),
            "correct": r.get("answer_correct"),
            "energy": r.get("energy_total_j"),
            "latency": r.get("total_latency_ms"),
            "ttft": r.get("ttft_ms"),
            "thinking_tokens": r.get("thinking_tokens"),
            "visible_output_tokens": r.get("visible_output_tokens"),
            "output_tokens": r.get("output_tokens"),
            "truncated": r.get("generation_truncated", False)
        })
    return pd.DataFrame(rows)
