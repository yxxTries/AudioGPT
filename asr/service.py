
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Optional
from .config import config
from .ffmpeg_io import normalize_to_wav
from .types import ASRResult
from .faster_whisper_backend import FasterWhisperBackend

class ASRService:
    def __init__(self, backend=None):
        self.backend = backend or FasterWhisperBackend()
        self.sample_rate = config.sample_rate

    def transcribe_file(self, input_path: str, *, language: Optional[str] = None, prompt: Optional[str] = None) -> ASRResult:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing audio: {input_path}")

        with tempfile.TemporaryDirectory() as tmp:
            norm = Path(tmp) / "normalized.wav"
            wav = normalize_to_wav(str(p), str(norm), sample_rate=self.sample_rate)
            result = self.backend.transcribe(wav, language=language, prompt=prompt)
            result.text = " ".join(result.text.split())
            return result
