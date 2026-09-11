"""Budget-constrained prompt strategy selector."""

from typing import List, Dict, Any, Optional


def select_prompt_strategy(
    strategy_metrics: List[Dict[str, Any]],
    accuracy_target: Optional[float] = None,
    energy_budget: Optional[float] = None,
    latency_budget: Optional[float] = None,
    optimization_priority: str = "energy"  # 'energy', 'accuracy', or 'latency'
) -> Optional[Dict[str, Any]]:
    """Selects the optimal prompting strategy meeting explicit operational budgets.
    
    Args:
        strategy_metrics: List of summary metric dicts for different strategies.
        accuracy_target: Minimum required accuracy (e.g., 0.70).
        energy_budget: Maximum permissible mean energy in Joules (e.g., 5.0).
        latency_budget: Maximum permissible mean latency in milliseconds (e.g., 1000.0).
        optimization_priority: Objective to optimize among feasible candidates.
    
    Returns:
        Best feasible strategy dict, or None if no strategy satisfies constraints.
    """
    feasible = []

    for strat in strategy_metrics:
        acc = strat.get("accuracy", 0.0)
        energy = strat.get("mean_energy_j")
        latency = strat.get("mean_latency_ms")

        # Check accuracy target
        if accuracy_target is not None and acc < accuracy_target:
            continue

        # Check energy budget
        if energy_budget is not None and (energy is None or energy > energy_budget):
            continue

        # Check latency budget
        if latency_budget is not None and (latency is None or latency > latency_budget):
            continue

        feasible.append(strat)

    if not feasible:
        return None

    # Optimize among feasible
    if optimization_priority == "energy":
        # Minimize energy, tie-break on accuracy
        return min(feasible, key=lambda s: (s.get("mean_energy_j") or float("inf"), -s.get("accuracy", 0.0)))
    elif optimization_priority == "accuracy":
        # Maximize accuracy, tie-break on energy
        return max(feasible, key=lambda s: (s.get("accuracy", 0.0), -(s.get("mean_energy_j") or float("inf"))))
    elif optimization_priority == "latency":
        # Minimize latency, tie-break on accuracy
        return min(feasible, key=lambda s: (s.get("mean_latency_ms") or float("inf"), -s.get("accuracy", 0.0)))
    else:
        return feasible[0]
