"""Unit tests for backend abstractions and reasoning token accounting."""

import unittest
from typing import List, Dict, Optional, Any
from src.backends.base import ModelBackend, InferenceOutput


class MockBackend(ModelBackend):
    def __init__(self, model_name: str = "mock-model", simulate_thinking: bool = False, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.simulate_thinking = simulate_thinking

    def load_model(self) -> None:
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
        if self.simulate_thinking:
            thinking = "Step 1: 2+2=4."
            response = "#### 4"
            raw_output = f"<think>\n{thinking}\n</think>\n{response}"
            return InferenceOutput(
                text=response,
                raw_thinking=thinking,
                raw_response=response,
                raw_output=raw_output,
                input_tokens=15,
                thinking_tokens=10,
                visible_output_tokens=5,
                output_tokens=15,
                total_tokens=30,
                token_count_method="backend_usage",
                reasoning_measurement_method="backend_reported",
                ttft_ms=120.0,
                generation_latency_ms=300.0,
                total_latency_ms=420.0
            )
        else:
            response = "#### 4"
            return InferenceOutput(
                text=response,
                raw_thinking=None,
                raw_response=response,
                raw_output=response,
                input_tokens=15,
                thinking_tokens=None,
                visible_output_tokens=5,
                output_tokens=5,
                total_tokens=20,
                token_count_method="backend_usage",
                reasoning_measurement_method="unavailable",
                ttft_ms=100.0,
                generation_latency_ms=150.0,
                total_latency_ms=250.0
            )

    def get_model_metadata(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "context_limit": 4096}


class TestBackend(unittest.TestCase):

    def test_mock_backend_standard(self):
        backend = MockBackend(simulate_thinking=False)
        out = backend.generate(messages=[{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(out.text, "#### 4")
        self.assertIsNone(out.thinking_tokens)
        self.assertEqual(out.visible_output_tokens, 5)
        self.assertEqual(out.output_tokens, 5)
        self.assertEqual(out.total_tokens, 20)
        self.assertEqual(out.reasoning_measurement_method, "unavailable")

    def test_mock_backend_reasoning_accounting(self):
        backend = MockBackend(simulate_thinking=True)
        out = backend.generate(messages=[{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(out.text, "#### 4")
        self.assertEqual(out.thinking_tokens, 10)
        self.assertEqual(out.visible_output_tokens, 5)
        self.assertEqual(out.output_tokens, out.thinking_tokens + out.visible_output_tokens)
        self.assertEqual(out.total_tokens, out.input_tokens + out.output_tokens)
        self.assertEqual(out.reasoning_measurement_method, "backend_reported")
    def test_mock_backend_prompt_argument(self):
        backend = MockBackend(simulate_thinking=False)
        out1 = backend.generate(prompt="What is 2+2?")
        self.assertEqual(out1.text, "#### 4")
        out2 = backend.generate(prompt=[{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(out2.text, "#### 4")


if __name__ == "__main__":
    unittest.main()
