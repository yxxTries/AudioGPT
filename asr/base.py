
from __future__ import annotations
from typing import Protocol, Optional
from .types import ASRResult

class ASRBackend(Protocol):
    def transcribe(self, audio_path: str, *, language: Optional[str] = None, prompt: Optional[str] = None) -> ASRResult:
        ...
