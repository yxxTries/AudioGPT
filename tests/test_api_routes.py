import sys
from pathlib import Path

# Add project root so imports work when running test manually
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import create_app
import api.deps as deps

class DummyASR:
    def transcribe_file(self, p, **kw):
        return type("R", (), {"text": "dummy transcript"})

class DummyLLM:
    def generate(self, req):
        return type("R", (), {"text": f"LLM({req.text})"})

def test_routes():
    app = create_app()
    app.dependency_overrides[deps.get_asr_service] = lambda: DummyASR()
    app.dependency_overrides[deps.get_llm_service] = lambda: DummyLLM()
    c = TestClient(app)

    r = c.post("/api/llm", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["text"] == "LLM(hello)"

    r = c.post("/api/asr-llm", files={"audio": ("x.webm", b"123", "audio/webm")})
    assert r.status_code == 200
    assert r.json()["transcript"] == "dummy transcript"
    assert r.json()["text"] == "LLM(dummy transcript)"
