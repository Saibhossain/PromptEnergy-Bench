"""HuggingFace Transformers backend adapter/stub."""

import time
from typing import Dict, List, Optional, Any
from src.backends.base import ModelBackend, InferenceOutput
from src.data.context_builder import estimate_tokens


class TransformersBackend(ModelBackend):
    """Adapter for local Hugging Face Transformers models."""

    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype: str = "auto", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch packages are required for TransformersBackend. "
                "Install with 'pip install transformers torch'."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype
        )
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

        import torch
        if seed is not None:
            torch.manual_seed(seed)

        start_time = time.perf_counter()
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_token_count = inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens or 512,
            "do_sample": temperature > 0.0,
            "top_p": top_p
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        end_time = time.perf_counter()

        generated_ids = output_ids[0][input_token_count:]
        output_tokens = len(generated_ids)
        total_tokens = input_token_count + output_tokens

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        total_latency_ms = (end_time - start_time) * 1000.0

        return InferenceOutput(
            text=text,
            raw_response=text,
            raw_output=text,
            input_tokens=input_token_count,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_count_method="model_tokenizer",
            reasoning_measurement_method="unavailable",
            ttft_ms=None,
            generation_latency_ms=round(total_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            backend_metadata={"model": self.model_name, "backend": "transformers"}
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "transformers",
            "context_limit": 4096
        }
