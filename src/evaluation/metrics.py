"""Statistical, Quality, and Green AI Metric Computation for PromptEnergy-Bench.

Implements the complete empirical framework from research_paper.md:
- Task-agnostic quality metric aggregation (Math, Multiple-Choice, Classification, Open-QA, Summarization, RAG, Code, Safety).
- Energy phase decompositions: E_total = E_prefill + E_decode + E_overhead, E_RAG = E_embedding + E_retrieval + E_prefill + E_decode + E_overhead, E_net = E_system - E_idle.
- Marginal Computational Efficiency:
    * Marginal Energy Gain: MEG(B1, B2) = [A(B2) - A(B1)] / [E(B2) - E(B1)]
    * Marginal Accuracy per Token: MAG_token(B1, B2) = [A(B2) - A(B1)] / [Tokens(B2) - Tokens(B1)]
    * Marginal Accuracy per Latency: MAG_latency(B1, B2) = [A(B2) - A(B1)] / [Latency(B2) - Latency(B1)]
- Budget-Constrained Prompting Selection (solving max A(pi) s.t. E(pi) <= B_E, T(pi) <= B_T).
- Explicit sample lifecycle states, validity statuses, and zero data coercion.
"""

import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union

from src.evaluation.base import MetricValidityStatus, EvaluationStatus


def _is_valid_num(val: Any) -> bool:
    """Checks if a value is a valid, non-null, non-NaN float or integer."""
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def _safe_mean(vals: List[float], round_digits: int = 4) -> Optional[float]:
    """Computes mean over valid numbers, or returns None if list is empty."""
    valid = [v for v in vals if _is_valid_num(v)]
    if not valid:
        return None
    return round(statistics.mean(valid), round_digits)


def _safe_median(vals: List[float], round_digits: int = 4) -> Optional[float]:
    """Computes median over valid numbers, or returns None if list is empty."""
    valid = [v for v in vals if _is_valid_num(v)]
    if not valid:
        return None
    return round(statistics.median(valid), round_digits)


def _safe_stdev(vals: List[float], round_digits: int = 4) -> Optional[float]:
    """Computes standard deviation over valid numbers (returns 0.0 if n=1, None if empty)."""
    valid = [v for v in vals if _is_valid_num(v)]
    if not valid:
        return None
    if len(valid) == 1:
        return 0.0
    return round(statistics.stdev(valid), round_digits)


def calculate_meg(
    acc_base: Optional[float],
    acc_candidate: Optional[float],
    energy_base: Optional[float],
    energy_candidate: Optional[float]
) -> Optional[float]:
    """Calculates Marginal Energy Gain (MEG).
    
    MEG(B1, B2) = [A(B2) - A(B1)] / [E(B2) - E(B1)]
    """
    if (
        not _is_valid_num(acc_base) or
        not _is_valid_num(acc_candidate) or
        not _is_valid_num(energy_base) or
        not _is_valid_num(energy_candidate)
    ):
        return None

    delta_energy = energy_candidate - energy_base
    delta_acc = acc_candidate - acc_base

    if abs(delta_energy) < 1e-7:
        return None
    return round(delta_acc / delta_energy, 6)


def calculate_mag_token(
    acc_base: Optional[float],
    acc_candidate: Optional[float],
    tokens_base: Optional[float],
    tokens_candidate: Optional[float]
) -> Optional[float]:
    """Calculates Marginal Accuracy per Token (MAG_token).
    
    MAG_token(B1, B2) = [A(B2) - A(B1)] / [Tokens(B2) - Tokens(B1)]
    """
    if (
        not _is_valid_num(acc_base) or
        not _is_valid_num(acc_candidate) or
        not _is_valid_num(tokens_base) or
        not _is_valid_num(tokens_candidate)
    ):
        return None

    delta_tokens = tokens_candidate - tokens_base
    delta_acc = acc_candidate - acc_base

    if abs(delta_tokens) < 1e-7:
        return None
    return round(delta_acc / delta_tokens, 6)


def calculate_mag_latency(
    acc_base: Optional[float],
    acc_candidate: Optional[float],
    latency_base: Optional[float],
    latency_candidate: Optional[float]
) -> Optional[float]:
    """Calculates Marginal Accuracy per Latency (MAG_latency).
    
    MAG_latency(B1, B2) = [A(B2) - A(B1)] / [Latency(B2) - Latency(B1)]
    """
    if (
        not _is_valid_num(acc_base) or
        not _is_valid_num(acc_candidate) or
        not _is_valid_num(latency_base) or
        not _is_valid_num(latency_candidate)
    ):
        return None

    # Assume latency in ms, convert difference to seconds
    delta_sec = (latency_candidate - latency_base) / 1000.0
    delta_acc = acc_candidate - acc_base

    if abs(delta_sec) < 1e-7:
        return None
    return round(delta_acc / delta_sec, 6)


def solve_budget_constrained_prompting(
    candidates: List[Dict[str, Any]],
    max_energy_j: Optional[float] = None,
    max_latency_ms: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Identifies optimal prompting strategy under explicit energy and latency budgets.
    
    Solves: max_pi A(pi) s.t. E(pi) <= B_E and T(pi) <= B_T
    """
    valid_candidates = []
    for c in candidates:
        acc = c.get("accuracy") if _is_valid_num(c.get("accuracy")) else c.get("total_accuracy")
        energy = c.get("energy_j") if _is_valid_num(c.get("energy_j")) else c.get("mean_energy_j")
        lat = c.get("latency_ms") if _is_valid_num(c.get("latency_ms")) else c.get("mean_total_latency_ms")

        if not _is_valid_num(acc):
            continue

        # Check energy constraint
        if max_energy_j is not None:
            if not _is_valid_num(energy) or energy > max_energy_j:
                continue

        # Check latency constraint
        if max_latency_ms is not None:
            if not _is_valid_num(lat) or lat > max_latency_ms:
                continue

        valid_candidates.append({**c, "_acc": acc, "_energy": energy or 0.0, "_lat": lat or 0.0})

    if not valid_candidates:
        return None

    # Sort descending by accuracy, ascending by energy, ascending by latency
    valid_candidates.sort(key=lambda x: (-x["_acc"], x["_energy"], x["_lat"]))
    best = valid_candidates[0]
    best.pop("_acc", None)
    best.pop("_energy", None)
    best.pop("_lat", None)
    return best


def compute_strategy_summary(records: List[Dict[str, Any]], task_type: Optional[str] = None) -> Dict[str, Any]:
    """Computes comprehensive Green AI and quality metrics for a single prompting condition."""
    total_requested = len(records)
    if total_requested == 0:
        return {}

    successful = [r for r in records if r.get("status") == "success"]
    total_successful = len(successful)
    total_failed = total_requested - total_successful

    # Sample lifecycle counts
    truncated_count = sum(
        1 for r in records
        if r.get("generation_truncated") is True or
        r.get("generation_stop_reason") == "length" or
        r.get("evaluation_status") == EvaluationStatus.GENERATION_TRUNCATED.value
    )

    # Valid evaluation samples: completed, not truncated, parseable
    valid_samples = [
        r for r in successful
        if not (
            r.get("generation_truncated") is True or
            r.get("generation_stop_reason") == "length" or
            r.get("evaluation_status") == EvaluationStatus.GENERATION_TRUNCATED.value
        )
        and (r.get("answer_parse_success") is True or r.get("parse_success") is True)
    ]
    valid_count = len(valid_samples)
    invalid_count = total_requested - valid_count

    parse_success_count = sum(
        1 for r in successful
        if (r.get("answer_parse_success") is True or r.get("parse_success") is True) and not (
            r.get("generation_truncated") is True or
            r.get("generation_stop_reason") == "length" or
            r.get("evaluation_status") == EvaluationStatus.GENERATION_TRUNCATED.value
        )
    )

    # Explicit evaluation status breakdown
    status_counts: Dict[str, int] = {}
    for r in records:
        st = r.get("evaluation_status")
        if st:
            status_counts[st] = status_counts.get(st, 0) + 1

    # Quality metrics
    correct_count = sum(1 for r in successful if r.get("answer_correct") is True)
    exact_match_count = sum(
        1 for r in successful
        if r.get("exact_match") is True or (isinstance(r.get("metric_values"), dict) and r["metric_values"].get("exact_match") is True)
    )

    # Denominator metrics
    total_accuracy = round(correct_count / total_requested, 4) if total_requested > 0 else 0.0
    valid_accuracy = round(correct_count / valid_count, 4) if valid_count > 0 else None
    exact_match_accuracy = round(exact_match_count / total_requested, 4) if total_requested > 0 else 0.0

    coverage = round(valid_count / total_requested, 4) if total_requested > 0 else 0.0
    parse_success_rate = round(parse_success_count / total_requested, 4) if total_requested > 0 else 0.0
    truncation_rate = round(truncated_count / total_requested, 4) if total_requested > 0 else 0.0
    completion_rate = round(total_successful / total_requested, 4) if total_requested > 0 else 0.0
    failure_rate = round(total_failed / total_requested, 4) if total_requested > 0 else 0.0

    # Metric validity status
    if total_requested == 0:
        acc_validity = MetricValidityStatus.NO_REFERENCE_ANSWERS
    elif valid_count == 0:
        acc_validity = MetricValidityStatus.INSUFFICIENT_VALID_SAMPLES
    else:
        acc_validity = MetricValidityStatus.COMPUTABLE

    # Task-specific metric distributions (F1, ROUGE, BLEU, Pass rate, Refusal rate)
    f1_list, rouge1_list, rouge2_list, rougeL_list, bleu_list, pass_list, refusal_list = [], [], [], [], [], [], []
    for r in valid_samples:
        mv = r.get("metric_values") or {}
        if _is_valid_num(mv.get("token_f1")):
            f1_list.append(float(mv["token_f1"]))
        if _is_valid_num(mv.get("rouge1_f1")):
            rouge1_list.append(float(mv["rouge1_f1"]))
        if _is_valid_num(mv.get("rouge2_f1")):
            rouge2_list.append(float(mv["rouge2_f1"]))
        if _is_valid_num(mv.get("rougeL_f1")):
            rougeL_list.append(float(mv["rougeL_f1"]))
        if _is_valid_num(mv.get("bleu")):
            bleu_list.append(float(mv["bleu"]))
        if _is_valid_num(mv.get("pass_rate")):
            pass_list.append(float(mv["pass_rate"]))
        if mv.get("is_refusal") is True:
            refusal_list.append(1.0)
        elif mv.get("is_refusal") is False:
            refusal_list.append(0.0)

    # Latencies
    gen_lats = [r["generation_latency_ms"] for r in successful if _is_valid_num(r.get("generation_latency_ms"))]
    tot_lats = [r["total_latency_ms"] for r in successful if _is_valid_num(r.get("total_latency_ms"))]
    ttfts = [r["ttft_ms"] for r in successful if _is_valid_num(r.get("ttft_ms"))]

    # Tokens (Strictly avoiding substituting configured limits)
    in_tokens = [r["input_tokens"] for r in successful if _is_valid_num(r.get("input_tokens"))]
    thinking_tokens = [r["thinking_tokens"] for r in successful if _is_valid_num(r.get("thinking_tokens"))]
    visible_tokens = [r["visible_output_tokens"] for r in successful if _is_valid_num(r.get("visible_output_tokens"))]
    output_tokens = [r["output_tokens"] for r in successful if _is_valid_num(r.get("output_tokens"))]
    total_tokens = [r["total_tokens"] for r in successful if _is_valid_num(r.get("total_tokens"))]

    # Energy (Preserving None for unmeasured fields)
    energies = [r["energy_total_j"] for r in successful if _is_valid_num(r.get("energy_total_j"))]
    prefill_energies = [r["energy_prefill_j"] for r in successful if _is_valid_num(r.get("energy_prefill_j"))]
    decode_energies = [r["energy_decode_j"] for r in successful if _is_valid_num(r.get("energy_decode_j"))]
    embedding_energies = [r["energy_embedding_j"] for r in successful if _is_valid_num(r.get("energy_embedding_j"))]
    retrieval_energies = [r["energy_retrieval_j"] for r in successful if _is_valid_num(r.get("energy_retrieval_j"))]
    net_energies = [r["energy_net_j"] for r in successful if _is_valid_num(r.get("energy_net_j"))]
    idle_powers = [r["idle_power_w"] for r in successful if _is_valid_num(r.get("idle_power_w"))]

    total_energy_j = round(sum(energies), 4) if energies else None
    mean_energy_j = _safe_mean(energies, 4)

    # Throughput: tokens per second
    tps_list = []
    for r in successful:
        toks = r.get("output_tokens")
        lat_ms = r.get("generation_latency_ms") or r.get("total_latency_ms")
        if _is_valid_num(toks) and _is_valid_num(lat_ms) and lat_ms > 0:
            tps_list.append(toks / (lat_ms / 1000.0))
    tokens_per_second = _safe_mean(tps_list, 2)

    # Efficiency metrics with safe denominators
    sum_out_tokens = sum(output_tokens) if output_tokens else 0
    sum_gen_lat = sum(gen_lats) if gen_lats else 0.0

    energy_per_output_token_j = (
        round(total_energy_j / sum_out_tokens, 6)
        if (total_energy_j is not None and sum_out_tokens > 0)
        else None
    )

    latency_per_output_token_ms = (
        round(sum_gen_lat / sum_out_tokens, 2)
        if (sum_gen_lat > 0 and sum_out_tokens > 0)
        else None
    )

    energy_per_correct_answer_j = (
        round(total_energy_j / correct_count, 4)
        if (total_energy_j is not None and correct_count > 0)
        else None
    )

    accuracy_per_joule = (
        round(total_accuracy / mean_energy_j, 4)
        if (total_accuracy is not None and mean_energy_j is not None and mean_energy_j > 0)
        else None
    )

    return {
        # Sample lifecycle counts
        "requested_samples": total_requested,
        "requested_inference_runs": total_requested,
        "n": total_requested,
        "completed_samples": total_successful,
        "successful_samples": total_successful,
        "successful_inference_runs": total_successful,
        "successful_n": total_successful,
        "failed_samples": total_failed,
        "failed_inference_runs": total_failed,
        "failed_n": total_failed,
        "truncated_samples": truncated_count,
        "truncated_generations": truncated_count,
        "truncated_n": truncated_count,
        "valid_samples": valid_count,
        "valid_n": valid_count,
        "invalid_samples": invalid_count,
        "invalid_n": invalid_count,
        "parseable_samples": parse_success_count,
        "parse_success_count": parse_success_count,
        "parse_success_n": parse_success_count,
        "correct_samples": correct_count,
        "correct_answers": correct_count,
        "correct_count": correct_count,
        "exact_match_answers": exact_match_count,
        "exact_match_count": exact_match_count,
        "status_distribution": status_counts,

        # Rates and Denominator Metrics
        "accuracy": total_accuracy,
        "total_accuracy": total_accuracy,
        "valid_accuracy": valid_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "coverage": coverage,
        "parse_success_rate": parse_success_rate,
        "truncation_rate": truncation_rate,
        "completion_rate": completion_rate,
        "failure_rate": failure_rate,
        "metric_validity_status": acc_validity.value,

        # Task Specific Quality Metrics
        "mean_token_f1": _safe_mean(f1_list, 4),
        "mean_rouge1_f1": _safe_mean(rouge1_list, 4),
        "mean_rouge2_f1": _safe_mean(rouge2_list, 4),
        "mean_rougeL_f1": _safe_mean(rougeL_list, 4),
        "mean_bleu": _safe_mean(bleu_list, 4),
        "mean_pass_rate": _safe_mean(pass_list, 4),
        "refusal_rate": _safe_mean(refusal_list, 4),

        # Latency statistics (ms)
        "mean_total_latency_ms": _safe_mean(tot_lats, 2),
        "median_total_latency_ms": _safe_median(tot_lats, 2),
        "std_total_latency_ms": _safe_stdev(tot_lats, 2),
        "mean_generation_latency_ms": _safe_mean(gen_lats, 2),
        "median_generation_latency_ms": _safe_median(gen_lats, 2),
        "std_generation_latency_ms": _safe_stdev(gen_lats, 2),
        "mean_ttft_ms": _safe_mean(ttfts, 2),
        "median_ttft_ms": _safe_median(ttfts, 2),
        "std_ttft_ms": _safe_stdev(ttfts, 2),

        # Token statistics
        "mean_input_tokens": _safe_mean(in_tokens, 2),
        "mean_thinking_tokens": _safe_mean(thinking_tokens, 2),
        "median_thinking_tokens": _safe_median(thinking_tokens, 2),
        "mean_visible_output_tokens": _safe_mean(visible_tokens, 2),
        "mean_output_tokens": _safe_mean(output_tokens, 2),
        "median_output_tokens": _safe_median(output_tokens, 2),
        "mean_total_tokens": _safe_mean(total_tokens, 2),

        # Energy statistics (Joules & Watts)
        "total_energy_j": total_energy_j,
        "mean_energy_j": mean_energy_j,
        "median_energy_j": _safe_median(energies, 4),
        "std_energy_j": _safe_stdev(energies, 4),
        "energy_per_sample_j": mean_energy_j,
        "mean_prefill_energy_j": _safe_mean(prefill_energies, 4),
        "mean_decode_energy_j": _safe_mean(decode_energies, 4),
        "mean_embedding_energy_j": _safe_mean(embedding_energies, 4),
        "mean_retrieval_energy_j": _safe_mean(retrieval_energies, 4),
        "mean_net_energy_j": _safe_mean(net_energies, 4),
        "mean_idle_power_w": _safe_mean(idle_powers, 4),

        # Throughput and Green AI Efficiency
        "tokens_per_second": tokens_per_second,
        "energy_per_output_token_j": energy_per_output_token_j,
        "latency_per_output_token_ms": latency_per_output_token_ms,
        "energy_per_correct_answer_j": energy_per_correct_answer_j,
        "accuracy_per_joule": accuracy_per_joule
    }


def compute_experiment_metrics(records: List[Dict[str, Any]], task_type: Optional[str] = None) -> Dict[str, Any]:
    """Computes full benchmark metrics aggregated across all records and per-strategy conditions."""
    if not records:
        return {}

    overall_summary = compute_strategy_summary(records, task_type=task_type)

    # Compute per-strategy breakdowns
    strategy_summaries: Dict[str, Any] = {}
    strategies_seen = sorted(list(set(r.get("strategy") for r in records if r.get("strategy"))))
    for strat in strategies_seen:
        strat_recs = [r for r in records if r.get("strategy") == strat]
        strategy_summaries[strat] = compute_strategy_summary(strat_recs, task_type=task_type)

    # Compute comparative marginal metrics (MEG, MAG_token, MAG_latency) relative to baseline
    baseline_strat = "zero_shot_direct" if "zero_shot_direct" in strategy_summaries else (
        strategies_seen[0] if strategies_seen else None
    )

    marginal_efficiency: Dict[str, Any] = {}
    if baseline_strat and baseline_strat in strategy_summaries:
        base = strategy_summaries[baseline_strat]
        b_acc = base.get("total_accuracy")
        b_eng = base.get("mean_energy_j")
        b_tok = base.get("mean_output_tokens")
        b_lat = base.get("mean_total_latency_ms")

        for strat in strategies_seen:
            if strat == baseline_strat:
                continue
            cand = strategy_summaries[strat]
            c_acc = cand.get("total_accuracy")
            c_eng = cand.get("mean_energy_j")
            c_tok = cand.get("mean_output_tokens")
            c_lat = cand.get("mean_total_latency_ms")

            meg = calculate_meg(b_acc, c_acc, b_eng, c_eng)
            mag_tok = calculate_mag_token(b_acc, c_acc, b_tok, c_tok)
            mag_lat = calculate_mag_latency(b_acc, c_acc, b_lat, c_lat)

            marginal_efficiency[strat] = {
                "baseline": baseline_strat,
                "meg_pct_per_joule": meg,
                "mag_pct_per_token": mag_tok,
                "mag_pct_per_second": mag_lat
            }

    # Merge overall results with strategy summaries and marginal efficiency
    result = dict(overall_summary)
    result["strategy_summaries"] = strategy_summaries
    result["marginal_efficiency"] = marginal_efficiency
    return result
