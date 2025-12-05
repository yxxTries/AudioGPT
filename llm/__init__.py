"""LLM package providing a simple interface to a local Qwen model."""

from .config import LLMConfig, DEFAULT_LLM_CONFIG
from .types import LLMRequest, LLMResponse
from .service import LLMService

__all__ = [
    "LLMConfig",
    "DEFAULT_LLM_CONFIG",
    "LLMRequest",
    "LLMResponse",
    "LLMService",
]
