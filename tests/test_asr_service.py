import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from asr.service import ASRService

AUDIO_SAMPLE_DIR = PROJECT_ROOT / "audiosample"

def test_asr():
    # Find audio file in audiosample folder
    audio_files = list(AUDIO_SAMPLE_DIR.glob("*.*"))
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {AUDIO_SAMPLE_DIR}")
    
    audio_path = audio_files[0]
    print(f"[TEST] Transcribing: {audio_path.name}")
    
    s = ASRService()
    r = s.transcribe_file(str(audio_path))
    
    # Check that we got some transcription
    assert r.text, "Transcription returned empty text"
    assert len(r.text) > 0, "Transcription text is empty"
    
    print(f"[TEST] Transcription: {r.text}")
    print(f"[TEST] Language: {r.language}")
    print(f"[TEST] Duration: {r.duration}s")

# test_asr()