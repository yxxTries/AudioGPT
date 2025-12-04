# src/app/main_asr_demo.py

import argparse
import os
from pathlib import Path

from asr.model import ASRModel
import re


def main():

    audio_path = 'src/app/audio/amil.m4a'
    model_size = "medium.en"

    if not os.path.isfile(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return

    # Instantiate ASR model
    asr = ASRModel(model_size=model_size)

    # Transcribe
    result = asr.transcribe_file(audio_path)
    # Clean and optimize the transcription text for LLM input
    text = result["text"].strip()
    # Remove excessive whitespace and normalize spacing
    text = " ".join(text.split())
    # Remove any potential artifacts or repeated punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    result["text"] = text

    # Print the results
    print("\n========== TRANSCRIPTION RESULT ==========")
    print(result["text"])
    print("==========================================\n")

    # Write output to a text file named output.txt (overwrite each run)
    output_path = Path(__file__).parent / "output.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"[INFO] Transcription written to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}")


if __name__ == "__main__":
    main()
