from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    """Request payload for the LLM service.

    Attributes:
        text: Input text that should be processed by the LLM.
        max_new_tokens: Optional override for maximum number of response tokens.
        temperature: Optional override for sampling temperature.
        top_p: Optional override for nucleus sampling probability mass.
        repetition_penalty: Optional override for repetition penalty.
        prompt_key: Optional prompt template key (see config.PROMPT_TEMPLATES).
    """

    text: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    prompt_key: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Response payload from the LLM service.

    Attributes:
        text: Text produced by the LLM.
    """

    text: str
