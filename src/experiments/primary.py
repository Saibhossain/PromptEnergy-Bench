"""Experiment 1: Standard Prompting vs Few-Shot vs Chain-of-Thought."""

import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.data.gsm8k import load_gsm8k
from src.prompts.prompt_registry import (
    PromptStrategy,
    format_gsm8k_prompt,
    get_prompt_hash
)
from src.evaluation import get_evaluator, EvaluationStatus
from src.infrastructure.checkpoint import compute_condition_key


class PrimaryExperiment(BaseExperiment):
    """Orchestrates Experiment 1."""

    def __init__(self, cli_args: Optional[Dict[str, Any]] = None, interactive: bool = False):
        super().__init__(
            experiment_name="primary_exp_gsm8k",
            cli_args=cli_args,
            interactive=interactive
        )

        self.strategies = [
            PromptStrategy.ZERO_SHOT_DIRECT,
            PromptStrategy.FEW_SHOT_3,
            PromptStrategy.ZERO_SHOT_COT,
            PromptStrategy.SHORT_COT,
            PromptStrategy.LONG_COT
        ]

        # Record prompt hash in config
        self.config["strategies"] = [s.value for s in self.strategies]
        self.config["prompt_hash"] = get_prompt_hash()
        self.evaluator = get_evaluator(self.config.get("dataset", {}).get("name", "gsm8k"))

    def run(self) -> Dict[str, Any]:
        self.logger.info("Loading GSM8K evaluation dataset (TEST split)...")
        eval_records = load_gsm8k(split="test", eval_size=self.eval_size)
        self.logger.info(f"Loaded {len(eval_records)} evaluation examples.")

        # Warmup with standard zero-shot direct prompt
        warmup_prompt = format_gsm8k_prompt(
            question="If John has 5 apples and eats 2, how many apples does he have left?",
            strategy=PromptStrategy.ZERO_SHOT_DIRECT
        )
        self.run_warmup(warmup_prompt)

        # Main Evaluation factorial loop
        total_eval_steps = len(self.strategies) * self.repetitions * len(eval_records)
        self.logger.info(f"Beginning main evaluation across {total_eval_steps} inference requests...")

        pbar = tqdm(total=total_eval_steps, desc="Evaluation Progress")

        for strategy in self.strategies:
            strat_max_tokens = self.get_max_tokens_for_strategy(strategy.value)

            for rep in range(1, self.repetitions + 1):
                for sample in eval_records:
                    cond_key = compute_condition_key(
                        experiment_name=self.experiment_name,
                        sample_id=sample.id,
                        model=self.model_name,
                        strategy=strategy.value,
                        repetition=rep
                    )

                    if self.checkpoint_mgr.is_completed(cond_key):
                        pbar.update(1)
                        continue

                    # Construct exact prompt
                    prompt_text = format_gsm8k_prompt(
                        question=sample.question,
                        strategy=strategy
                    )

                    sampling_cfg = self.config.get("sampling", {})
                    temperature = sampling_cfg.get("temperature", 0.0)
                    seed = sampling_cfg.get("seed", 42)
                    top_p = sampling_cfg.get("top_p", 1.0)

                    self.energy_monitor.start(phase="total")
                    self.resource_monitor.start()
                    status = "success"
                    error_type = None
                    error_message = None

                    try:
                        infer_out = self.backend.generate(
                            messages=prompt_text,
                            temperature=temperature,
                            max_tokens=strat_max_tokens,
                            seed=seed,
                            top_p=top_p,
                            stream=True
                        )
                    except Exception as e:
                        status = "failed"
                        error_type = type(e).__name__
                        error_message = str(e)
                        infer_out = None

                    resource_reading = self.resource_monitor.stop()
                    energy_reading = self.energy_monitor.stop(phase="total")

                    if infer_out is not None:
                        is_truncated = bool(
                            infer_out.generation_truncated or
                            infer_out.generation_stop_reason == "length"
                        )
                        eval_res = self.evaluator.evaluate(
                            raw_output=infer_out.text,
                            gold_answer=sample.answer,
                            raw_response=infer_out.raw_response,
                            generation_truncated=is_truncated,
                            sample_id=sample.id
                        )
                        
                        record_error_type = "generation_truncated" if is_truncated else None

                        record = {
                            "run_id": self.run_id,
                            "experiment_name": self.experiment_name,
                            "sample_id": sample.id,
                            "task_type": self.evaluator.config.task_type.value if hasattr(self.evaluator.config.task_type, "value") else str(self.evaluator.config.task_type),
                            "model": self.model_name,
                            "backend": self.operator,
                            "strategy": strategy.value,
                            "repetition": rep,
                            "condition_key": cond_key,
                            "status": status,
                            
                            # Universal evaluation fields
                            "reference_answer": sample.answer,
                            "raw_response": infer_out.raw_response,
                            "normalized_response": eval_res.normalized_response,
                            "parsed_answer": eval_res.parsed_answer,
                            "evaluation_status": eval_res.evaluation_status.value if hasattr(eval_res.evaluation_status, "value") else str(eval_res.evaluation_status),
                            "answer_correct": eval_res.answer_correct,
                            "metric_values": eval_res.metric_values,
                            "parse_success": eval_res.parse_success,
                            "generation_truncated": is_truncated,
                            "generation_stop_reason": infer_out.generation_stop_reason,
                            "generation_complete": infer_out.generation_complete and not is_truncated,

                            # Token counts
                            "actual_token_counts": {
                                "configured_max_tokens": strat_max_tokens,
                                "actual_output_tokens": infer_out.output_tokens,
                                "actual_thinking_tokens": infer_out.thinking_tokens,
                                "actual_visible_tokens": infer_out.visible_output_tokens,
                                "total_generated_tokens": infer_out.total_tokens
                            },
                            
                            # Latency metrics
                            "latency_metrics": {
                                "ttft_ms": infer_out.ttft_ms,
                                "generation_latency_ms": infer_out.generation_latency_ms,
                                "total_latency_ms": infer_out.total_latency_ms
                            },
                            
                            # Energy metrics
                            "energy_metrics": {
                                "energy_total_j": energy_reading.energy_total_j,
                                "energy_prefill_j": energy_reading.energy_prefill_j,
                                "energy_decode_j": energy_reading.energy_decode_j,
                                "energy_embedding_j": energy_reading.energy_embedding_j,
                                "energy_retrieval_j": energy_reading.energy_retrieval_j,
                                "energy_overhead_j": energy_reading.energy_overhead_j,
                                "energy_net_j": energy_reading.energy_net_j,
                                "idle_power_w": energy_reading.idle_power_w,
                                "energy_status": energy_reading.energy_status
                            },
                            
                            # Resource metrics (CPU, RAM, GPU)
                            "resource_metrics": resource_reading.to_dict(),
                            
                            # Legacy flat fields for backward compatibility
                            "max_output_tokens": strat_max_tokens,
                            "thinking_text_available": infer_out.thinking_text_available,
                            "input_tokens": infer_out.input_tokens,
                            "thinking_tokens": infer_out.thinking_tokens,
                            "visible_output_tokens": infer_out.visible_output_tokens,
                            "output_tokens": infer_out.output_tokens,
                            "total_tokens": infer_out.total_tokens,
                            "token_count_method": infer_out.token_count_method,
                            "reasoning_measurement_method": infer_out.reasoning_measurement_method,
                            "ttft_ms": infer_out.ttft_ms,
                            "generation_latency_ms": infer_out.generation_latency_ms,
                            "total_latency_ms": infer_out.total_latency_ms,
                            "cpu_percent": resource_reading.cpu_percent_mean,
                            "cpu_percent_peak": resource_reading.cpu_percent_peak,
                            "ram_used_gb": resource_reading.ram_used_gb_mean,
                            "ram_percent": resource_reading.ram_percent,
                            "energy_total_j": energy_reading.energy_total_j,
                            "energy_prefill_j": energy_reading.energy_prefill_j,
                            "energy_decode_j": energy_reading.energy_decode_j,
                            "energy_embedding_j": energy_reading.energy_embedding_j,
                            "energy_retrieval_j": energy_reading.energy_retrieval_j,
                            "energy_overhead_j": energy_reading.energy_overhead_j,
                            "energy_net_j": energy_reading.energy_net_j,
                            "idle_power_w": energy_reading.idle_power_w,
                            "active_power_w": energy_reading.active_power_w,
                            "energy_measurement_method": energy_reading.energy_measurement_method,
                            "energy_quality": energy_reading.energy_quality,
                            "energy_measurement_level": energy_reading.energy_measurement_level,
                            "energy_status": energy_reading.energy_status,
                            "gold_answer": sample.answer,
                            "raw_thinking": infer_out.raw_thinking,
                            "raw_response": infer_out.raw_response,
                            "raw_output": infer_out.raw_output,
                            "extracted_answer": eval_res.extracted_answer,
                            "answer_parse_success": eval_res.answer_parse_success,
                            "exact_match": eval_res.exact_match,
                            "error_type": record_error_type,
                            "error_message": None,
                            # 26 Research Reproducibility Fields (Section 2.B)
                            "experiment_id": self.run_id,
                            "dataset": "gsm8k",
                            "dataset_split": "test",
                            "dataset_version": "main",
                            "question_id": sample.id,
                            "prompt_strategy": strategy.value,
                            "prompt_version": "v1.0",
                            "model_name": self.model_name,
                            "model_version": getattr(self.backend, "model_version", "default"),
                            "model_quantization": self.format,
                            "inference_engine": self.operator,
                            "hardware_identifier": self.normalized_device,
                            "operating_system": self.device_info.get("os", "unknown"),
                            "generation_parameters": {"temperature": 0.0, "seed": int(self.cli_args.get("seed", 42)), "max_tokens": strat_max_tokens, "top_p": 1.0},
                            "random_seed": int(self.cli_args.get("seed", 42)),
                            "input_token_count": infer_out.input_tokens,
                            "output_token_count": infer_out.output_tokens,
                            "reasoning_token_count": infer_out.thinking_tokens,
                            "decode_latency_ms": infer_out.generation_latency_ms,
                            "energy_measurement_units": "Joules",
                            "raw_energy_j": energy_reading.energy_total_j,
                            "idle_energy_j": round(energy_reading.idle_power_w * (infer_out.total_latency_ms / 1000.0), 4) if (energy_reading.idle_power_w and infer_out.total_latency_ms) else None,
                            "timestamp": self.timestamp,
                            "raw_model_output": infer_out.raw_output,
                            "extracted_prediction": eval_res.extracted_answer,
                            "correctness": eval_res.answer_correct,
                            "error_category": eval_res.error_type or ("none" if eval_res.answer_correct else "unknown_error")
                        }
                    else:
                        record = {
                            "run_id": self.run_id,
                            "experiment_name": self.experiment_name,
                            "sample_id": sample.id,
                            "task_type": self.evaluator.config.task_type.value if hasattr(self.evaluator.config.task_type, "value") else str(self.evaluator.config.task_type),
                            "model": self.model_name,
                            "backend": self.operator,
                            "strategy": strategy.value,
                            "repetition": rep,
                            "condition_key": cond_key,
                            "status": "failed",
                            "reference_answer": sample.answer,
                            "raw_response": None,
                            "normalized_response": None,
                            "parsed_answer": None,
                            "evaluation_status": EvaluationStatus.EVALUATOR_ERROR.value,
                            "answer_correct": False,
                            "metric_values": {},
                            "parse_success": False,
                            "generation_stop_reason": "error",
                            "generation_truncated": False,
                            "generation_complete": False,
                            "actual_token_counts": {
                                "configured_max_tokens": strat_max_tokens,
                                "actual_output_tokens": 0,
                                "actual_thinking_tokens": None,
                                "actual_visible_tokens": 0,
                                "total_generated_tokens": 0
                            },
                            "latency_metrics": {
                                "ttft_ms": None,
                                "generation_latency_ms": None,
                                "total_latency_ms": None
                            },
                            "energy_metrics": {
                                "energy_total_j": None,
                                "energy_prefill_j": None,
                                "energy_decode_j": None,
                                "energy_embedding_j": None,
                                "energy_retrieval_j": None,
                                "energy_overhead_j": None,
                                "energy_net_j": None,
                                "idle_power_w": None,
                                "energy_status": "unavailable"
                            },
                            "resource_metrics": resource_reading.to_dict(),
                            "max_output_tokens": strat_max_tokens,
                            "thinking_text_available": False,
                            "input_tokens": 0,
                            "thinking_tokens": None,
                            "visible_output_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "token_count_method": "unavailable",
                            "reasoning_measurement_method": "unavailable",
                            "ttft_ms": None,
                            "generation_latency_ms": None,
                            "total_latency_ms": None,
                            "cpu_percent": resource_reading.cpu_percent_mean,
                            "cpu_percent_peak": resource_reading.cpu_percent_peak,
                            "ram_used_gb": resource_reading.ram_used_gb_mean,
                            "ram_percent": resource_reading.ram_percent,
                            "energy_total_j": None,
                            "energy_prefill_j": None,
                            "energy_decode_j": None,
                            "energy_overhead_j": None,
                            "energy_net_j": None,
                            "idle_power_w": None,
                            "active_power_w": None,
                            "energy_measurement_method": energy_reading.energy_measurement_method,
                            "energy_quality": energy_reading.energy_quality,
                            "energy_measurement_level": energy_reading.energy_measurement_level,
                            "energy_status": "unavailable",
                            "gold_answer": sample.answer,
                            "raw_thinking": None,
                            "raw_response": None,
                            "raw_output": None,
                            "extracted_answer": None,
                            "answer_parse_success": False,
                            "answer_correct": False,
                            "error_type": error_type,
                            "error_message": error_message
                        }

                    self.checkpoint_mgr.record_result(record)
                    pbar.update(1)

        pbar.close()
        return self.finalize()
