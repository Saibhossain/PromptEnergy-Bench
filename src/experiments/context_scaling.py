"""Experiment 2: Controlled Context Scaling using GSM8K-derived examples."""

import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.data.gsm8k import load_gsm8k
from src.data.context_builder import ContextBuilder, ScaledContext
from src.prompts.gsm8k_prompts import PromptStrategy, format_gsm8k_prompt
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.infrastructure.checkpoint import compute_condition_key


class ContextScalingExperiment(BaseExperiment):
    """Orchestrates Experiment 2: Controlled Context Scaling."""

    def __init__(self, cli_args: Optional[Dict[str, Any]] = None, interactive: bool = False):
        super().__init__(
            experiment_name="context_scaling_gsm8k",
            cli_args=cli_args,
            interactive=interactive
        )

        self.context_sizes = [512, 1024, 2048, 4096]
        if self.cli_args.get("include_8k", False):
            self.context_sizes.append(8192)

        self.context_types = ["relevant", "distractor"]
        self.strategy = PromptStrategy.ZERO_SHOT_DIRECT

    def run(self) -> Dict[str, Any]:
        self.logger.info("Loading evaluation dataset (TEST split) and context corpus (TRAIN split)...")
        eval_records = load_gsm8k(split="test", eval_size=self.eval_size)
        train_records = load_gsm8k(split="train")
        context_builder = ContextBuilder(train_records=train_records)

        # Check model context limit
        model_meta = self.backend.get_model_metadata()
        model_ctx_limit = model_meta.get("context_limit", 4096)
        self.logger.info(f"Model context limit: {model_ctx_limit} tokens.")

        # Warmup
        warmup_msgs = format_gsm8k_prompt(self.strategy, "What is 2+2?")
        self.run_warmup(warmup_msgs)

        sampling = self.config["sampling"]
        temp = sampling["temperature"]
        seed = sampling["seed"]
        top_p = sampling["top_p"]
        max_tokens = sampling["max_tokens"]

        total_steps = len(eval_records) * len(self.context_types) * len(self.context_sizes) * self.repetitions
        pbar = tqdm(total=total_steps, desc="Context Scaling Progress")

        for sample in eval_records:
            for ctx_type in self.context_types:
                for ctx_size in self.context_sizes:
                    for rep in range(1, self.repetitions + 1):
                        cond_key = compute_condition_key(
                            experiment_name=self.experiment_name,
                            sample_id=sample.id,
                            model=self.model_name,
                            strategy=self.strategy.value,
                            repetition=rep,
                            context_type=ctx_type,
                            context_target_tokens=ctx_size
                        )

                        if self.checkpoint_mgr.is_completed(cond_key):
                            pbar.update(1)
                            continue

                        # Check if context exceeds model limit
                        if ctx_size > model_ctx_limit:
                            skip_record = {
                                "run_id": self.run_id,
                                "experiment_name": self.experiment_name,
                                "sample_id": sample.id,
                                "model": self.model_name,
                                "backend": self.operator,
                                "strategy": self.strategy.value,
                                "repetition": rep,
                                "condition_key": cond_key,
                                "context_source_type": ctx_type,
                                "context_target_tokens": ctx_size,
                                "actual_context_tokens": 0,
                                "context_document_ids": [],
                                "status": "skipped",
                                "reason": "context_limit",
                                "error_type": "ContextLimitExceeded",
                                "error_message": f"Requested tokens ({ctx_size}) > model context limit ({model_ctx_limit})"
                            }
                            self.checkpoint_mgr.record_result(skip_record)
                            pbar.update(1)
                            continue

                        # Build controlled context from GSM8K train split
                        scaled_ctx: ScaledContext = context_builder.build_context(
                            target_record=sample,
                            context_type=ctx_type,
                            target_tokens=ctx_size
                        )

                        messages = format_gsm8k_prompt(
                            strategy=self.strategy,
                            question=sample.question,
                            context=scaled_ctx.context_text
                        )

                        self.energy_monitor.start(phase="total")
                        error_type = None
                        error_message = None
                        status = "success"

                        try:
                            infer_out = self.backend.generate(
                                messages=messages,
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
                            eval_res = GSM8KEvaluator.evaluate(
                                raw_output=infer_out.text,
                                gold_answer=sample.answer
                            )
                            record = {
                                "run_id": self.run_id,
                                "experiment_name": self.experiment_name,
                                "sample_id": sample.id,
                                "model": self.model_name,
                                "backend": self.operator,
                                "strategy": self.strategy.value,
                                "repetition": rep,
                                "condition_key": cond_key,
                                "context_source_type": ctx_type,
                                "context_target_tokens": ctx_size,
                                "actual_context_tokens": scaled_ctx.actual_context_tokens,
                                "context_document_ids": scaled_ctx.context_document_ids,
                                "status": status,
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
                                "answer_correct": eval_res.answer_correct,
                                "error_type": None,
                                "error_message": None
                            }
                        else:
                            record = {
                                "run_id": self.run_id,
                                "experiment_name": self.experiment_name,
                                "sample_id": sample.id,
                                "model": self.model_name,
                                "backend": self.operator,
                                "strategy": self.strategy.value,
                                "repetition": rep,
                                "condition_key": cond_key,
                                "context_source_type": ctx_type,
                                "context_target_tokens": ctx_size,
                                "actual_context_tokens": scaled_ctx.actual_context_tokens,
                                "context_document_ids": scaled_ctx.context_document_ids,
                                "status": "failed",
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
