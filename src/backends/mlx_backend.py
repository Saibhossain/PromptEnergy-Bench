"""MLX-LM native backend adapter/stub for Apple Silicon."""

import time
from typing import Dict, List, Optional, Any
from src.backends.base import ModelBackend, InferenceOutput
from src.data.context_builder import estimate_tokens


class MLXBackend(ModelBackend):
    """Adapter for Apple Silicon native MLX-LM models."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        try:
            import mlx_lm
        except ImportError:
            raise ImportError(
                "mlx-lm is required for MLXBackend. "
                "Install with 'pip install mlx-lm'."
            )
        self.model, self.tokenizer = mlx_lm.load(self.model_name)
        self.is_loaded = True

    def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
        seed: Optional[int] = 42,
        top_p: float = 1.0,
        stream: bool = False,
        prompt: Optional[Any] = None
    ) -> InferenceOutput:
        messages = self._normalize_messages(messages, prompt)
        if not self.is_loaded:
            self.load_model()

        import mlx_lm
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(self.tokenizer.encode(prompt_text))

        start_time = time.perf_counter()
        response_text = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt_text,
            temp=temperature,
            max_tokens=max_tokens or 512,
            top_p=top_p,
            verbose=False
        )
        end_time = time.perf_counter()

        output_tokens = len(self.tokenizer.encode(response_text))
        total_latency_ms = (end_time - start_time) * 1000.0

        return InferenceOutput(
            text=response_text.strip(),
            raw_response=response_text.strip(),
            raw_output=response_text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            token_count_method="model_tokenizer",
            reasoning_measurement_method="unavailable",
            ttft_ms=None,
            generation_latency_ms=round(total_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            backend_metadata={"model": self.model_name, "backend": "mlx"}
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "mlx",
            "context_limit": 4096
        }
