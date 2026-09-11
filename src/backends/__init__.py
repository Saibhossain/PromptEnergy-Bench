"""Model backends abstraction layer."""
from src.backends.base import ModelBackend, InferenceOutput
from src.backends.ollama_backend import OllamaBackend
from src.backends.openai_backend import OpenAIBackend
from src.backends.transformers_backend import TransformersBackend
from src.backends.mlx_backend import MLXBackend
from src.backends.llama_cpp_backend import LlamaCppBackend

__all__ = [
    "ModelBackend",
    "InferenceOutput",
    "OllamaBackend",
    "OpenAIBackend",
    "TransformersBackend",
    "MLXBackend",
    "LlamaCppBackend"
]


def get_backend(operator: str, model_name: str, **kwargs) -> ModelBackend:
    """Factory to instantiate the appropriate ModelBackend."""
    op = operator.lower().strip()
    if op == "ollama":
        return OllamaBackend(model_name=model_name, **kwargs)
    elif op in ("openai", "api"):
        return OpenAIBackend(model_name=model_name, **kwargs)
    elif op in ("transformers", "hf"):
        return TransformersBackend(model_name=model_name, **kwargs)
    elif op == "mlx":
        return MLXBackend(model_name=model_name, **kwargs)
    elif op in ("llama.cpp", "llamacpp"):
        return LlamaCppBackend(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model operator/backend: '{operator}'. Supported: ollama, openai, transformers, mlx, llama.cpp")
