"""Ollama Model Backend with streaming TTFT and reasoning token accounting."""

import time
from typing import Dict, List, Optional, Any
import ollama

from src.backends.base import ModelBackend, InferenceOutput
from src.data.context_builder import estimate_tokens


class OllamaBackend(ModelBackend):
    """Ollama execution adapter.
    
    Supports local GGUF/MLX/FP16 models served through Ollama, capturing
    exact prompt/eval token metrics and streaming TTFT.
    """

    def __init__(self, model_name: str, host: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.host = host
        self.client = ollama.Client(host=host) if host else ollama
        self.model_info: Dict[str, Any] = {}

    def load_model(self) -> None:
        """Verifies that the requested model is available locally in Ollama."""
        try:
            available_models = self.client.list()
            # Extract names from models response
            model_names = []
            for m in getattr(available_models, "models", []):
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name:
                    model_names.append(name)

            # Match either exact name or prefix (e.g. qwen3.5:0.8b-mlx matching qwen3.5:0.8b-mlx:latest)
            matched = any(
                m == self.model_name or m.startswith(f"{self.model_name}:")
                for m in model_names
            )
            if not matched:
                raise RuntimeError(
                    f"Requested model '{self.model_name}' is not available in the selected backend.\n"
                    f"Available Ollama models: {model_names}\n"
                    f"Run 'ollama pull {self.model_name}' or verify spelling."
                )

            # Retrieve detailed model info
            try:
                show_info = self.client.show(self.model_name)
                self.model_info = {
                    "modelfile": getattr(show_info, "modelfile", None),
                    "parameters": getattr(show_info, "parameters", None),
                    "template": getattr(show_info, "template", None),
                    "details": getattr(show_info, "details", None)
                }
            except Exception:
                self.model_info = {"model": self.model_name}

            self.is_loaded = True
        except Exception as e:
            if "not available in the selected backend" in str(e):
                raise
            raise ConnectionError(
                f"Failed to connect to Ollama server at '{self.host or 'localhost:11434'}': {e}"
            ) from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
        seed: Optional[int] = 42,
        top_p: float = 1.0,
        stream: bool = True
    ) -> InferenceOutput:
        if not self.is_loaded:
            self.load_model()

        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p)
        }
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if seed is not None:
            options["seed"] = int(seed)

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None

        thinking_parts: List[str] = []
        content_parts: List[str] = []

        final_prompt_eval_count = 0
        final_eval_count = 0
        final_prompt_eval_duration_ns = 0
        final_eval_duration_ns = 0
        raw_done_reason: Optional[str] = None

        try:
            if stream:
                stream_resp = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=True
                )
                for chunk in stream_resp:
                    now = time.perf_counter()
                    dr = getattr(chunk, "done_reason", None)
                    if dr:
                        raw_done_reason = dr

                    msg = getattr(chunk, "message", None)
                    if msg:
                        # Check thinking attribute (e.g. Qwen 3.5 internal reasoning)
                        th = getattr(msg, "thinking", None)
                        if th:
                            if first_token_time is None:
                                first_token_time = now
                            thinking_parts.append(th)

                        ct = getattr(msg, "content", None)
                        if ct:
                            if first_token_time is None:
                                first_token_time = now
                            content_parts.append(ct)

                    # Update usage from final chunk
                    pec = getattr(chunk, "prompt_eval_count", None)
                    if pec is not None:
                        final_prompt_eval_count = pec
                    ec = getattr(chunk, "eval_count", None)
                    if ec is not None:
                        final_eval_count = ec
                    ped = getattr(chunk, "prompt_eval_duration", None)
                    if ped is not None:
                        final_prompt_eval_duration_ns = ped
                    ed = getattr(chunk, "eval_duration", None)
                    if ed is not None:
                        final_eval_duration_ns = ed
            else:
                resp = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=False
                )
                first_token_time = time.perf_counter()
                raw_done_reason = getattr(resp, "done_reason", None)
                msg = getattr(resp, "message", None)
                if msg:
                    th = getattr(msg, "thinking", None)
                    if th:
                        thinking_parts.append(th)
                    ct = getattr(msg, "content", None)
                    if ct:
                        content_parts.append(ct)
                final_prompt_eval_count = getattr(resp, "prompt_eval_count", 0) or 0
                final_eval_count = getattr(resp, "eval_count", 0) or 0
                final_prompt_eval_duration_ns = getattr(resp, "prompt_eval_duration", 0) or 0
                final_eval_duration_ns = getattr(resp, "eval_duration", 0) or 0

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000.0
            raise RuntimeError(f"Ollama generation error: {e}") from e

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else None
        generation_latency_ms = ((end_time - first_token_time) * 1000.0) if first_token_time else total_latency_ms

        raw_thinking_str = "".join(thinking_parts).strip()
        raw_response_str = "".join(content_parts).strip()

        # Build raw_output
        if raw_thinking_str:
            raw_output = f"<think>\n{raw_thinking_str}\n</think>\n{raw_response_str}"
            text = raw_response_str if raw_response_str else raw_output
            has_thinking = True
        else:
            raw_output = raw_response_str
            text = raw_response_str
            has_thinking = False

        # Stop reason and truncation normalization
        if raw_done_reason:
            norm_reason = str(raw_done_reason).lower().strip()
            if norm_reason in ("length", "max_tokens"):
                generation_stop_reason = "length"
                generation_truncated = True
                generation_complete = False
            elif norm_reason in ("stop", "end_of_turn"):
                generation_stop_reason = "stop"
                generation_truncated = False
                generation_complete = True
            elif norm_reason == "eos":
                generation_stop_reason = "eos"
                generation_truncated = False
                generation_complete = True
            elif "error" in norm_reason:
                generation_stop_reason = "error"
                generation_truncated = False
                generation_complete = False
            else:
                generation_stop_reason = norm_reason
                generation_truncated = None
                generation_complete = None
        else:
            generation_stop_reason = "unknown"
            generation_truncated = None
            generation_complete = None

        # Token accounting
        # Ollama's final_eval_count is total generated tokens (thinking + content)
        total_output_tokens = final_eval_count
        if total_output_tokens == 0:
            total_output_tokens = estimate_tokens(raw_output)

        if has_thinking:
            if raw_response_str == "" and generation_truncated is True:
                # Entire output budget was consumed inside thinking phase
                thinking_tokens = total_output_tokens
                visible_output_tokens = 0
                reasoning_measurement_method = "backend_reported"
            else:
                # Backend does not report separate token count for thinking vs content
                # Do NOT infer or estimate thinking token count from text length or latency
                thinking_tokens = None
                visible_output_tokens = total_output_tokens
                reasoning_measurement_method = "unavailable"
        else:
            thinking_tokens = None
            visible_output_tokens = total_output_tokens
            reasoning_measurement_method = "unavailable"

        input_tokens = final_prompt_eval_count
        if input_tokens == 0:
            # Fallback estimation if prompt eval count wasn't returned
            prompt_text = " ".join([m.get("content", "") for m in messages])
            input_tokens = estimate_tokens(prompt_text)

        total_tokens = input_tokens + total_output_tokens

        return InferenceOutput(
            text=text,
            raw_thinking=raw_thinking_str if has_thinking else None,
            raw_response=raw_response_str,
            raw_output=raw_output,
            input_tokens=input_tokens,
            thinking_tokens=thinking_tokens,
            visible_output_tokens=visible_output_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            token_count_method="backend_usage" if final_eval_count > 0 else "estimated",
            reasoning_measurement_method=reasoning_measurement_method,
            ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
            generation_latency_ms=round(generation_latency_ms, 2) if generation_latency_ms is not None else None,
            total_latency_ms=round(total_latency_ms, 2),
            generation_stop_reason=generation_stop_reason,
            generation_truncated=generation_truncated,
            generation_complete=generation_complete,
            max_output_tokens=int(max_tokens) if max_tokens is not None else 1024,
            thinking_text_available=has_thinking,
            backend_metadata={
                "prompt_eval_duration_ms": round(final_prompt_eval_duration_ns / 1e6, 2) if final_prompt_eval_duration_ns else None,
                "eval_duration_ms": round(final_eval_duration_ns / 1e6, 2) if final_eval_duration_ns else None,
                "raw_done_reason": raw_done_reason,
                "model": self.model_name
            }
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "ollama",
            "context_limit": 4096,  # Default Ollama context limit for standard runner
            "info": self.model_info
        }
