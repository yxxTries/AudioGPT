
import logging, subprocess
from pathlib import Path
from .exceptions import AudioPreprocessingError

logger = logging.getLogger(__name__)

def normalize_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise AudioPreprocessingError(f"Missing audio: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-ac", "1", "-ar", str(sample_rate), str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AudioPreprocessingError("ffmpeg failed") from exc

    return str(out_path)
