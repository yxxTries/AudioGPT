#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from pathlib import Path
from typing import Tuple

from asr.config import ASRConfig
from asr.service import ASRService
from asr.types import ASRResult

from llm.config import LLMConfig
from llm.service import LLMService
from llm.types import LLMRequest  


logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    """
    Configure root logging based on a simple -v / -vv flag scheme.
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(
    audio_path: Path,
    asr_service: ASRService,
    llm_service: LLMService,
    max_duration_s: float = 30.0,
) -> Tuple[ASRResult, str]:
    """
    Run the end-to-end pipeline:
    audio file -> ASR transcription -> LLM response.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe.
    asr_service:
        An initialized ASRService instance.
    llm_service:
        An initialized LLMService instance.
    max_duration_s:
        Expected maximum audio duration in seconds. Used for logging only.

    Returns
    -------
    (asr_result, llm_output)
    """
    logger.info("Starting ASR transcription for %s", audio_path)
    asr_result = asr_service.transcribe_file(audio_path)

    logger.info("Transcription completed. Detected duration: %.2fs", asr_result.duration)

    if asr_result.duration > max_duration_s:
        logger.warning(
            "Audio duration (%.2fs) exceeds recommended max_duration_s=%.2fs.",
            asr_result.duration,
            max_duration_s,
        )

    logger.debug("Transcribed text: %s", asr_result.text)

    logger.info("Sending transcription to LLM.")
    llm_output = llm_service.generate(LLMRequest(text=asr_result.text))
    logger.info("LLM generation completed.")

    return asr_result, llm_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simple CLI test that runs ASR + LLM on an audio file. "
            "Intended for voice recordings up to ~30 seconds."
        )
    )

    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to an audio file (recommended <= 30 seconds).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum expected audio duration in seconds (for logging only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    )

    return parser


def main() -> None:
    audio_path = Path("audiosample/recording.wav")
    max_duration = 30.0
    verbosity = 1  # INFO level logging

    setup_logging(verbosity)

    if not audio_path.is_file():
        print(f"Audio file does not exist: {audio_path}")
        return

    logger.info("Initializing ASR service.")
    asr_service = ASRService()

    logger.info("Initializing LLM service.")
    llm_config = LLMConfig()
    llm_service = LLMService(llm_config)

    asr_result, llm_output = run_pipeline(
        audio_path=audio_path,
        asr_service=asr_service,
        llm_service=llm_service,
        max_duration_s=max_duration,
    )

    print("=== Transcription ===")
    print(asr_result.text)
    print("\n=== LLM Response ===")
    print(llm_output)


if __name__ == "__main__":
    main()
