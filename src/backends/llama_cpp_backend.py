"""llama-cpp-python backend adapter/stub."""

import time
from typing import Dict, List, Optional, Any
from src.backends.base import ModelBackend, InferenceOutput
from src.data.context_builder import estimate_tokens


class LlamaCppBackend(ModelBackend):
    """Adapter for GGUF models executed via llama-cpp-python."""

    def __init__(self, model_name: str, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

    def load_model(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install with 'pip install llama-cpp-python'."
            )
        self.llm = Llama(
            model_path=self.model_name,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False
        )
        self.is_loaded = True

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
        seed: Optional[int] = 42,
        top_p: float = 1.0,
        stream: bool = False
    ) -> InferenceOutput:
        if not self.is_loaded:
            self.load_model()

        start_time = time.perf_counter()
        resp = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 512,
            top_p=top_p,
            seed=seed
        )
        end_time = time.perf_counter()

        text = resp["choices"][0]["message"]["content"].strip()
        usage = resp.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) or estimate_tokens(str(messages))
        output_tokens = usage.get("completion_tokens", 0) or estimate_tokens(text)
        total_tokens = input_tokens + output_tokens

        total_latency_ms = (end_time - start_time) * 1000.0

        return InferenceOutput(
            text=text,
            raw_response=text,
            raw_output=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_count_method="backend_usage",
            reasoning_measurement_method="unavailable",
            ttft_ms=None,
            generation_latency_ms=round(total_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            backend_metadata={"model": self.model_name, "backend": "llama.cpp"}
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "llama.cpp",
            "context_limit": self.n_ctx
        }
