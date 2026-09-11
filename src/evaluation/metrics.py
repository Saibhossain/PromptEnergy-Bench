"""Statistical and Green AI metric computation."""

import statistics
from typing import Dict, List, Any, Optional


def calculate_meg(
    acc_base: float,
    acc_candidate: float,
    energy_base: float,
    energy_candidate: float
) -> Optional[float]:
    """Calculates Marginal Energy Gain (MEG).
    
    MEG(B1, B2) = [A(B2) - A(B1)] / [E(B2) - E(B1)]
    Returns None if energy difference is zero or negative (handled explicitly).
    """
    delta_energy = energy_candidate - energy_base
    delta_acc = acc_candidate - acc_base

    if abs(delta_energy) < 1e-7:
        return None
    return delta_acc / delta_energy


def compute_experiment_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive Green AI metrics aggregated across records."""
    if not records:
        return {}

    successful = [r for r in records if r.get("status") == "success"]
    total_requested = len(records)
    total_successful = len(successful)
    total_failed = total_requested - total_successful

    if total_successful == 0:
        return {
            "requested_samples": total_requested,
            "successful_samples": 0,
            "failed_samples": total_failed,
            "accuracy": 0.0
        }

    correct_count = sum(1 for r in successful if r.get("answer_correct") is True)
    accuracy = correct_count / total_successful

    latencies = [r["total_latency_ms"] for r in successful if r.get("total_latency_ms") is not None]
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    output_tokens = [r["output_tokens"] for r in successful if r.get("output_tokens") is not None]
    thinking_tokens = [r["thinking_tokens"] for r in successful if r.get("thinking_tokens") is not None]
    visible_tokens = [r["visible_output_tokens"] for r in successful if r.get("visible_output_tokens") is not None]
    energies = [r["energy_total_j"] for r in successful if r.get("energy_total_j") is not None]

    mean_latency = statistics.mean(latencies) if latencies else None
    median_latency = statistics.median(latencies) if latencies else None

    mean_ttft = statistics.mean(ttfts) if ttfts else None

    mean_tokens = statistics.mean(output_tokens) if output_tokens else None
    median_tokens = statistics.median(output_tokens) if output_tokens else None

    mean_thinking_tokens = statistics.mean(thinking_tokens) if thinking_tokens else None
    mean_visible_tokens = statistics.mean(visible_tokens) if visible_tokens else None

    mean_energy = statistics.mean(energies) if energies else None
    median_energy = statistics.median(energies) if energies else None
    total_energy = sum(energies) if energies else None

    energy_per_correct = (total_energy / correct_count) if (total_energy and correct_count > 0) else None
    sum_tokens = sum(output_tokens) if output_tokens else 0
    energy_per_token = (total_energy / sum_tokens) if (total_energy and sum_tokens > 0) else None
    accuracy_per_joule = (accuracy / mean_energy) if (mean_energy and mean_energy > 0) else None

    return {
        "requested_samples": total_requested,
        "successful_samples": total_successful,
        "failed_samples": total_failed,
        "correct_answers": correct_count,
        "accuracy": round(accuracy, 4),
        "mean_latency_ms": round(mean_latency, 2) if mean_latency else None,
        "median_latency_ms": round(median_latency, 2) if median_latency else None,
        "mean_ttft_ms": round(mean_ttft, 2) if mean_ttft else None,
        "mean_output_tokens": round(mean_tokens, 2) if mean_tokens else None,
        "median_output_tokens": median_tokens,
        "mean_thinking_tokens": round(mean_thinking_tokens, 2) if mean_thinking_tokens else None,
        "mean_visible_output_tokens": round(mean_visible_tokens, 2) if mean_visible_tokens else None,
        "mean_energy_j": round(mean_energy, 4) if mean_energy else None,
        "median_energy_j": round(median_energy, 4) if median_energy else None,
        "total_energy_j": round(total_energy, 4) if total_energy else None,
        "energy_per_correct_answer_j": round(energy_per_correct, 4) if energy_per_correct else None,
        "energy_per_output_token_j": round(energy_per_token, 6) if energy_per_token else None,
        "accuracy_per_joule": round(accuracy_per_joule, 4) if accuracy_per_joule else None
    }
