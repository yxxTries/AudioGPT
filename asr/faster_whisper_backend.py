
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, List
from faster_whisper import WhisperModel
from .types import ASRResult, ASRSegment
from .exceptions import ASRModelError
from .config import config

logger = logging.getLogger(__name__)

@dataclass
class FasterWhisperBackend:
    model_name: str = config.model_name
    device: str = config.device
    compute_type: str = config.compute_type

    def __post_init__(self):
        try:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        except Exception as exc:
            raise ASRModelError("Failed to load model") from exc

    def transcribe(self, audio_path: str, *, language: Optional[str] = None, prompt: Optional[str] = None) -> ASRResult:
        try:
            segments_iter, info = self._model.transcribe(
                audio_path,
                language=language or config.language,
                beam_size=config.beam_size,
                vad_filter=config.vad_filter,
                initial_prompt=prompt,
            )
        except Exception as exc:
            raise ASRModelError("Transcription failed") from exc

        segments: List[ASRSegment] = []
        texts: List[str] = []

        for seg in segments_iter:
            t = seg.text.strip()
            if t:
                segments.append(ASRSegment(float(seg.start), float(seg.end), t))
                texts.append(t)

        return ASRResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", None),
            segments=segments,
            duration=getattr(info, "duration", None),
        )

    def transcribe_audio(self, audio_data, sample_rate: int = 16000, *, language: Optional[str] = None, prompt: Optional[str] = None) -> ASRResult:
        """Transcribe from numpy array audio data directly (no file needed)."""
        import numpy as np
        
        # Ensure audio is float32 and normalized for faster_whisper
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Resample if needed (faster_whisper expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
        
        try:
            segments_iter, info = self._model.transcribe(
                audio_data,
                language=language or config.language,
                beam_size=config.beam_size,
                vad_filter=config.vad_filter,
                initial_prompt=prompt,
            )
        except Exception as exc:
            raise ASRModelError("Transcription failed") from exc

        segments: List[ASRSegment] = []
        texts: List[str] = []

        for seg in segments_iter:
            t = seg.text.strip()
            if t:
                segments.append(ASRSegment(float(seg.start), float(seg.end), t))
                texts.append(t)

        return ASRResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", None),
            segments=segments,
            duration=getattr(info, "duration", None),
        )
