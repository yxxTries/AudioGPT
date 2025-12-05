from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the local Qwen LLM.

    Attributes:
        model_dir: Path to the directory containing the Qwen model weights and tokenizer.
        device: Device identifier understood by the underlying ML framework (e.g. "cpu", "cuda").
        max_new_tokens: Default maximum number of tokens to generate for a response.
        temperature: Default sampling temperature for generation.
    """

    model_dir: Path = Path("models") / "models--Qwen--Qwen2-0.5B-Instruct" / "snapshots" / "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    device: str = "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.7


DEFAULT_LLM_CONFIG = LLMConfig()
