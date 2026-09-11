"""Analysis package: statistical testing, Pareto frontier, and budget selection."""
from src.analysis.statistics import (
    calculate_summary_statistics,
    compute_confidence_interval,
    paired_wilcoxon_test
)
from src.analysis.pareto import find_pareto_frontier, is_pareto_efficient
from src.analysis.selection import select_prompt_strategy

__all__ = [
    "calculate_summary_statistics",
    "compute_confidence_interval",
    "paired_wilcoxon_test",
    "find_pareto_frontier",
    "is_pareto_efficient",
    "select_prompt_strategy"
]
