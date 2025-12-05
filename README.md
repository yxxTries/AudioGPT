# AudioGPT

A local voice-to-LLM application that transcribes audio using Whisper and generates responses with a local Qwen model.

## Project Structure

```
AudioGPT/
├── api/          # FastAPI server (REST endpoints)
├── asr/          # Automatic Speech Recognition (faster-whisper)
├── llm/          # Local LLM service (Qwen)
├── ui/           # Web frontend
├── tests/        # Test files
└── models/       # Local model files
```

## Requirements

```txt
fastapi>=0.123.0
uvicorn>=0.38.0
python-multipart>=0.0.20
pydantic-settings>=2.12.0
faster-whisper>=1.2.1
transformers>=4.57.0
torch>=2.9.0
sounddevice>=0.5.3
numpy>=2.3.0
scipy>=1.16.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yxxTries/AudioGPT.git
cd AudioGPT

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Running

### Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at `http://localhost:8000`

### Start the UI

```bash
cd ui
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

### CLI Voice Conversation

```bash
python tests/test_voice_conversation_cli.py
```

Press ENTER to start recording, ENTER again to stop.

### API Documentation

Once running, access the interactive docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

- `POST /api/llm` - Send text to LLM
- `POST /api/asr-llm` - Upload audio for transcription + LLM response
