from __future__ import annotations

import logging
from typing import Any

from .config import LLMConfig, DEFAULT_LLM_CONFIG
from .model_loader import QwenModelLoader
from .types import LLMRequest, LLMResponse


logger = logging.getLogger(__name__)


class LLMService:
    """High-level service to generate responses from a local Qwen model.

    This service is intentionally minimal: it accepts plain text input (e.g. ASR output)
    and returns a plain text response suitable for feeding back into a UI layer.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._config = config or DEFAULT_LLM_CONFIG

        if model is None or tokenizer is None:
            loader = QwenModelLoader(self._config.model_dir, device=self._config.device)
            tokenizer, model = loader.load()

        self._model = model
        self._tokenizer = tokenizer

        logger.info(
            "LLMService initialized with model_dir=%s, device=%s",
            self._config.model_dir,
            self._config.device,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given request.

        Args:
            request: Request object containing the input text and optional parameters.

        Returns:
            LLMResponse with the generated text.
        """
        prompt = request.text
        if not prompt:
            raise ValueError("LLMRequest.text must not be empty")

        max_new_tokens = request.max_new_tokens or self._config.max_new_tokens
        temperature = request.temperature if request.temperature is not None else self._config.temperature

        logger.debug(
            "Generating response (max_new_tokens=%s, temperature=%s)", max_new_tokens, temperature
        )

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Generate output token ids
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )

        # Decode the full sequence
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the original prompt if the model echoes it back
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text

        logger.debug("Generated response length=%d characters", len(generated_text))

        return LLMResponse(text=generated_text)

    def generate_from_text(
        self,
        text: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Convenience wrapper that accepts raw text instead of an LLMRequest.

        This is the method you are most likely to call from the ASR/UI pipeline:

            response = llm_service.generate_from_text(transcribed_text)
        """
        request = LLMRequest(text=text, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.generate(request)
