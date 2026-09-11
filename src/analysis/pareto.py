"""Pareto efficiency and dominance analysis."""

from typing import List, Dict, Any


def is_pareto_efficient(
    candidate: Dict[str, Any],
    all_configs: List[Dict[str, Any]],
    maximize_keys: List[str] = None,
    minimize_keys: List[str] = None
) -> bool:
    """Tests if a candidate configuration is Pareto-efficient (non-dominated).
    
    A configuration is dominated if another has equal or better in all objectives
    and strictly better in at least one objective.
    """
    maximize_keys = maximize_keys or ["accuracy"]
    minimize_keys = minimize_keys or ["mean_energy_j", "mean_latency_ms"]

    c_max = [candidate.get(k, 0.0) for k in maximize_keys]
    c_min = [candidate.get(k, float("inf")) for k in minimize_keys]

    for other in all_configs:
        if other is candidate:
            continue

        o_max = [other.get(k, 0.0) for k in maximize_keys]
        o_min = [other.get(k, float("inf")) for k in minimize_keys]

        # Check if 'other' is at least as good in all objectives
        better_or_equal_max = all(o >= c for o, c in zip(o_max, c_max))
        better_or_equal_min = all(o <= c for o, c in zip(o_min, c_min))

        # Check if 'other' is strictly better in at least one objective
        strictly_better_max = any(o > c for o, c in zip(o_max, c_max))
        strictly_better_min = any(o < c for o, c in zip(o_min, c_min))

        if better_or_equal_max and better_or_equal_min and (strictly_better_max or strictly_better_min):
            # candidate is dominated by 'other'
            return False

    return True


def find_pareto_frontier(
    configs: List[Dict[str, Any]],
    maximize_keys: List[str] = None,
    minimize_keys: List[str] = None
) -> List[Dict[str, Any]]:
    """Returns all Pareto-efficient configurations."""
    maximize_keys = maximize_keys or ["accuracy"]
    minimize_keys = minimize_keys or ["mean_energy_j", "mean_latency_ms"]

    frontier = []
    for cfg in configs:
        if is_pareto_efficient(cfg, configs, maximize_keys=maximize_keys, minimize_keys=minimize_keys):
            cfg_copy = dict(cfg)
            cfg_copy["is_pareto"] = True
            frontier.append(cfg_copy)
    return frontier
