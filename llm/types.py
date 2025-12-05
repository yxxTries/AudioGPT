from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    """Request payload for the LLM service.

    Attributes:
        text: Input text that should be processed by the LLM.
        max_new_tokens: Optional override for maximum number of response tokens.
        temperature: Optional override for sampling temperature.
    """

    text: str
    max_new_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Response payload from the LLM service.

    Attributes:
        text: Text produced by the LLM.
    """

    text: str
