"""Base ModelBackend interface and standardized inference output dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class InferenceOutput:
    text: str                                    # Normalized final text output
    raw_thinking: Optional[str] = None          # Thinking / internal reasoning content if exposed
    raw_response: str = ""                       # Visible text response
    raw_output: str = ""                         # Complete raw output (thinking + response)
    
    # Token accounting
    input_tokens: int = 0
    thinking_tokens: Optional[int] = None
    visible_output_tokens: Optional[int] = None
    output_tokens: int = 0
    total_tokens: int = 0
    token_count_method: str = "unavailable"      # backend_usage, model_tokenizer, estimated, unavailable
    reasoning_measurement_method: str = "unavailable" # backend_reported, unavailable
    
    # Latencies in milliseconds
    ttft_ms: Optional[float] = None              # Time to first token
    generation_latency_ms: Optional[float] = None # Time from first token to end of generation
    total_latency_ms: float = 0.0                # Total inference duration
    
    # Generation completion & stop metadata
    generation_stop_reason: str = "unknown"       # stop, length, eos, error, unknown
    generation_truncated: Optional[bool] = None  # True if stopped due to max tokens, False if normal, None if unknown
    generation_complete: Optional[bool] = None   # True if completed normally
    max_output_tokens: int = 1024
    thinking_text_available: bool = False

    # Backend metadata
    backend_metadata: Dict[str, Any] = field(default_factory=dict)


class ModelBackend(ABC):
    """Abstract Model Backend interface.
    
    Ensures complete isolation of experiment logic from specific execution providers
    (Ollama, OpenAI, Transformers, MLX, llama.cpp).
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Loads or validates the model."""
        pass

    def _normalize_messages(
        self,
        messages: Optional[Any] = None,
        prompt: Optional[Any] = None
    ) -> List[Dict[str, str]]:
        """Normalizes messages/prompt input into a List[Dict[str, str]]."""
        if messages is None and prompt is not None:
            messages = prompt
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return messages
        if messages is None:
            return [{"role": "user", "content": ""}]
        return messages

    @abstractmethod
    def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
        seed: Optional[int] = 42,
        top_p: float = 1.0,
        stream: bool = True,
        prompt: Optional[Any] = None
    ) -> InferenceOutput:
        """Executes generation against the model and returns standardized InferenceOutput."""
        pass

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the loaded model (e.g. context limit, format, size)."""
        pass

    def supports_token_usage(self) -> bool:
        return True

    def supports_ttft(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_energy_measurement(self) -> bool:
        return False

    def unload_model(self) -> None:
        """Unloads model resources if supported."""
        self.is_loaded = False
