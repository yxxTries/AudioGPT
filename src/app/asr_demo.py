# src/app/main_asr_demo.py

import argparse
import os
from pathlib import Path

from asr.model import ASRModel


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
