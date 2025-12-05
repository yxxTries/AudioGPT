from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple


logger = logging.getLogger(__name__)


class QwenModelLoader:
    """Loads a local Qwen model and tokenizer from disk.

    This loader expects the model to be available in a directory that is compatible
    with `transformers.AutoTokenizer.from_pretrained` and
    `transformers.AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self._model_dir = Path(model_dir)
        self._device = device

    def load(self) -> Tuple[Any, Any]:
        """Load the tokenizer and model.

        Returns:
            A `(tokenizer, model)` tuple.

        Raises:
            RuntimeError: If the `transformers` library is not installed.
            FileNotFoundError: If the model directory does not exist.
        """
        if not self._model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self._model_dir}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - exercised only in real runtime
            raise RuntimeError(
                "The 'transformers' package is required to load the Qwen model. "
                "Install it via 'pip install transformers'."
            ) from exc

        logger.info("Loading Qwen model from %s", self._model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self._model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self._model_dir, trust_remote_code=True)

        if hasattr(model, "to") and self._device:
            logger.info("Moving model to device: %s", self._device)
            model = model.to(self._device)

        logger.info("Qwen model loaded successfully")
        return tokenizer, model
