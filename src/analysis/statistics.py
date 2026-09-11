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


def paired_wilcoxon_test(
    sample_a: List[float],
    sample_b: List[float]
) -> Dict[str, Any]:
    """Runs Wilcoxon signed-rank test for paired continuous measurements."""
    if len(sample_a) != len(sample_b) or len(sample_a) < 5:
        return {"statistic": None, "p_value": None, "valid": False, "note": "Insufficient paired samples"}

    diffs = [a - b for a, b in zip(sample_a, sample_b)]
    if all(d == 0 for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0, "valid": True, "note": "Identical paired samples"}

    try:
        res = stats.wilcoxon(sample_a, sample_b)
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "significant_005": bool(res.pvalue < 0.05),
            "valid": True
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "valid": False, "error": str(e)}
