# src/asr/model.py

from typing import Optional, Dict, Any

import whisper


class ASRModel:
    """
    Simple wrapper around a Whisper ASR model.

    Responsibilities:
    - Load a Whisper model of a given size (e.g. 'tiny', 'base', 'small.en').
    - Provide a method to transcribe audio files into text.
    """

    def __init__(self, model_size: str = "small.en", device: Optional[str] = None):
        """
        :param model_size: Whisper model size, e.g. 'tiny', 'base', 'small.en', 'medium.en', ...
                          Suffix '.en' uses English-optimized models.
        :param device: 'cpu' or 'cuda'. If None, Whisper chooses automatically.
        """
        self.model_size = model_size
        print(f"[ASR] Loading Whisper model '{model_size}' ...")
        self.model = whisper.load_model(model_size, device=device)
        print(f"[ASR] Model loaded.")

    def transcribe_file(self, audio_path: str, **whisper_options: Any) -> Dict[str, Any]:
        """
        Transcribe an audio file into text.

        :param audio_path: Path to the audio file (wav, mp3, m4a, etc.).
        :param whisper_options: Additional options passed to whisper.transcribe,
                                e.g. language="en", temperature=0.0, fp16=False, etc.
        :return: Full Whisper result dict. You can access:
                 - result['text'] for the transcript
                 - result['segments'] for segment-level info
        """
        print(f"[ASR] Transcribing file: {audio_path}")
        # Provide some sensible defaults; user can override via whisper_options
        default_options = {
            "language": "en",
            "fp16": False,  # useful on CPU; set True if using GPU
        }
        options = {**default_options, **whisper_options}

        result = self.model.transcribe(audio_path, **options)
        print(f"[ASR] Transcription finished.")
        return result
