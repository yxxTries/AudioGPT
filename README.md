# AudioGPT (Quick README)

This repo contains a simple audio→text pipeline using Whisper for ASR and a local Hugging Face model for follow‑up text processing.


## Requirements

### Python Dependencies
```bash
pip install faster-whisper transformers torch sounddevice keyboard scipy numpy huggingface_hub pytest
```

### System Dependencies
- **ffmpeg** - Required for audio processing. Install via:
  - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### ASR Module Dependencies
| Package | Purpose |
|---------|---------|
| `faster-whisper` | Fast Whisper ASR backend |
| `numpy` | Audio array processing |

### LLM Module Dependencies
| Package | Purpose |
|---------|---------|
| `transformers` | Hugging Face model loading |
| `torch` | PyTorch backend |
| `huggingface_hub` | Model downloading |

### Audio Recording Dependencies
| Package | Purpose |
|---------|---------|
| `sounddevice` | Microphone recording |
| `keyboard` | Keyboard input (requires admin on Windows) |
| `scipy` | WAV file writing |

### Testing
| Package | Purpose |
|---------|---------|
| `pytest` | Running tests |


## How to run:

After installing dependencies, download the LLM model:
```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct', cache_dir='models'))"
```

Run the ASR demo (requires admin for keyboard):
```bash
python src/app/asr_demo.py
```

Run tests:
```bash
python -m pytest tests/
```

