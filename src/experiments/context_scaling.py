"""Experiment 2: Controlled Context Scaling using GSM8K-derived examples."""

import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.data.gsm8k import load_gsm8k
from src.data.context_builder import ContextBuilder, ScaledContext
from src.prompts.prompt_registry import PromptStrategy, format_gsm8k_prompt
from src.evaluation import get_evaluator, EvaluationStatus
from src.infrastructure.checkpoint import compute_condition_key


class ContextScalingExperiment(BaseExperiment):
    """Orchestrates Experiment 2: Controlled Context Scaling."""

    def __init__(self, cli_args: Optional[Dict[str, Any]] = None, interactive: bool = False):
        super().__init__(
            experiment_name="context_scaling_gsm8k",
            cli_args=cli_args,
            interactive=interactive
        )

        if self.cli_args.get("context_lengths"):
            self.context_lengths = [int(x) for x in self.cli_args["context_lengths"]]
        else:
            self.context_lengths = [0, 512, 1024, 2048, 4096]
            if self.cli_args.get("include_8k", False):
                self.context_lengths.append(8192)

        self.strategy = PromptStrategy.ZERO_SHOT_COT
        self.config["context_lengths"] = self.context_lengths
        self.config["strategy"] = self.strategy.value
        self.evaluator = get_evaluator(self.config.get("dataset", {}).get("name", "gsm8k"))

    def run(self) -> Dict[str, Any]:
        self.logger.info("Loading GSM8K Train split for Context Corpus...")
        train_records = load_gsm8k(split="train")
        context_builder = ContextBuilder(train_records)

        self.logger.info("Loading GSM8K Test split for Evaluation...")
        eval_records = load_gsm8k(split="test", eval_size=self.eval_size)

        # Warmup
        warmup_msgs = format_gsm8k_prompt(
            question="What is 2+2?",
            strategy=self.strategy
        )
        self.run_warmup(warmup_msgs)

        sampling = self.config["sampling"]
        temp = sampling["temperature"]
        seed = sampling["seed"]
        top_p = sampling["top_p"]
        max_tokens = self.get_max_tokens_for_strategy(self.strategy.value)

        total_steps = len(eval_records) * len(self.context_lengths) * self.repetitions
        pbar = tqdm(total=total_steps, desc="Context Scaling Progress")

        for ctx_size in self.context_lengths:
            for rep in range(1, self.repetitions + 1):
                for sample in eval_records:
                    cond_key = compute_condition_key(
                        experiment_name=self.experiment_name,
                        sample_id=sample.id,
                        model=self.model_name,
                        strategy=f"ctx_{ctx_size}",
                        repetition=rep,
                        context_type="gsm8k_train_corpus" if ctx_size > 0 else "none",
                        context_target_tokens=ctx_size
                    )

                    if self.checkpoint_mgr.is_completed(cond_key):
                        pbar.update(1)
                        continue

                    scaled_ctx = context_builder.build_context(
                        target_tokens=ctx_size,
                        exclude_id=sample.id
                    )

                    ctx_type = "empty" if ctx_size == 0 else "gsm8k_train_corpus"

                    prompt_text = format_gsm8k_prompt(
                        question=sample.question,
                        strategy=self.strategy,
                        context=scaled_ctx.context_text,
                        allow_context=True
                    )

                    self.energy_monitor.start(phase="total")
                    error_type = None
                    error_message = None
                    status = "success"

                    try:
                        infer_out = self.backend.generate(
                            messages=prompt_text,
                            temperature=temp,
                            max_tokens=max_tokens,
                            seed=seed,
                            top_p=top_p,
                            stream=True
                        )
                    except Exception as e:
                        status = "failed"
                        error_type = type(e).__name__
                        error_message = str(e)
                        infer_out = None

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
                            "strategy": f"ctx_{ctx_size}",
                            "repetition": rep,
                            "condition_key": cond_key,
                            "context_source_type": ctx_type,
                            "context_target_tokens": ctx_size,
                            "actual_context_tokens": scaled_ctx.actual_context_tokens,
                            "context_document_ids": scaled_ctx.context_document_ids,
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
                                "configured_max_tokens": max_tokens,
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

                            # Legacy flat fields
                            "max_output_tokens": max_tokens,
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
                            "energy_total_j": energy_reading.energy_total_j,
                            "energy_prefill_j": energy_reading.energy_prefill_j,
                            "energy_decode_j": energy_reading.energy_decode_j,
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
                            "error_message": None
                        }
                    else:
                        record = {
                            "run_id": self.run_id,
                            "experiment_name": self.experiment_name,
                            "sample_id": sample.id,
                            "task_type": self.evaluator.config.task_type.value if hasattr(self.evaluator.config.task_type, "value") else str(self.evaluator.config.task_type),
                            "model": self.model_name,
                            "backend": self.operator,
                            "strategy": f"ctx_{ctx_size}",
                            "repetition": rep,
                            "condition_key": cond_key,
                            "context_source_type": ctx_type,
                            "context_target_tokens": ctx_size,
                            "actual_context_tokens": scaled_ctx.actual_context_tokens,
                            "context_document_ids": scaled_ctx.context_document_ids,
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
                                "configured_max_tokens": max_tokens,
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
                            "max_output_tokens": max_tokens,
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
