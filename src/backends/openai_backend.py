"""OpenAI / API Model Backend adapter."""

import os
import time
from typing import Dict, List, Optional, Any
from src.backends.base import ModelBackend, InferenceOutput
from src.data.context_builder import estimate_tokens


class OpenAIBackend(ModelBackend):
    """Adapter for OpenAI API and OpenAI-compatible endpoints."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def load_model(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required for OpenAIBackend.")
        self.is_loaded = True

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

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAIBackend. Install with 'pip install openai'.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        chunks_collected: List[str] = []

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed

        input_tokens = 0
        output_tokens = 0

        if stream:
            params["stream_options"] = {"include_usage": True}
            response = client.chat.completions.create(**params)
            for chunk in response:
                now = time.perf_counter()
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = now
                    chunks_collected.append(chunk.choices[0].delta.content)
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
        else:
            response = client.chat.completions.create(**params)
            first_token_time = time.perf_counter()
            content = response.choices[0].message.content or ""
            chunks_collected.append(content)
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else None
        gen_latency_ms = ((end_time - first_token_time) * 1000.0) if first_token_time else total_latency_ms

        raw_output = "".join(chunks_collected).strip()
        if output_tokens == 0:
            output_tokens = estimate_tokens(raw_output)
        if input_tokens == 0:
            prompt_str = " ".join([m.get("content", "") for m in messages])
            input_tokens = estimate_tokens(prompt_str)

        return InferenceOutput(
            text=raw_output,
            raw_response=raw_output,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            token_count_method="backend_usage" if input_tokens > 0 else "estimated",
            reasoning_measurement_method="unavailable",
            ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
            generation_latency_ms=round(gen_latency_ms, 2) if gen_latency_ms else None,
            total_latency_ms=round(total_latency_ms, 2),
            backend_metadata={"model": self.model_name, "backend": "openai"}
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "openai",
            "context_limit": 128000
        }
