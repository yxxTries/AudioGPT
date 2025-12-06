from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the local TinyLlama LLM.

    Attributes:
        model_dir: Path to the directory containing the TinyLlama model weights and tokenizer.
        device: Device identifier understood by the underlying ML framework (e.g. "cpu", "cuda").
        max_new_tokens: Default maximum number of tokens to generate for a response.
        temperature: Default sampling temperature for generation.
        top_p: Nucleus sampling probability mass to consider.
        repetition_penalty: Penalty applied to repeated tokens; 1.0 disables it.
    """

    model_dir: Path = Path("models") / "TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"  # "auto" picks CUDA if available, else CPU
    max_new_tokens: int = 50  # reduced for faster CPU inference
    temperature: float = 0.8  # slightly creative but coherent
    top_p: float = 0.92  # good balance for TinyLlama
    top_k: int = 40  # limits vocabulary for more focused responses
    repetition_penalty: float = 1.15  # TinyLlama needs stronger penalty to avoid loops
    system_prompt: str = "You are a concise AI. Give short, direct answers. Never repeat the user's question. Never say 'As an AI' or similar phrases."
    instruction_prompt: str = "Only 1-2 sentence answers are allowed. Be brief and to the point."
    load_in_8bit: bool = False
    load_in_4bit: bool = True  # enabled for GPU users, ignored on CPU
    cpu_dtype: str = "float32"  # used when running on CPU to avoid slow float16 emulation
    cuda_dtype: str = "float16"  # used when running on CUDA


DEFAULT_LLM_CONFIG = LLMConfig()

# Simple prompt templates that can be swapped/edited later.
# TinyLlama uses the Zephyr/Llama chat format
PROMPT_TEMPLATES = {
    "default": "<|system|>\nYou are an assistant that will help with tasks by answer questions. Only provide 1-2 sentence answers.</s>\n<|user|>\n{input}</s>\n<|assistant|>\n"
}

# Default prompt key to use when none is specified elsewhere.
DEFAULT_PROMPT_KEY = "default"
