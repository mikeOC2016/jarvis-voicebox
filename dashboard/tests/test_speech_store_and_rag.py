"""
Tests for audit P0 #4 (close arbitrary-TTS-proxy hole) and P1 #7 (RAG synthesis).

Decision (docs/rule23/round4_fixes/DECISIONS.md, codex+perplexity converge):
  Q2 -> server-side speech store + speech_id. /api/search-chat synthesizes the
        spoken text server-side and returns an opaque speech_id; /api/jarvis/speak
        accepts speech_id ONLY (no raw caller text).
  Q1 -> RAG synthesis defaults to a LOCAL Ollama brain (Rule 36 PGX-LOCAL-FIRST),
        selectable via JARVIS_BRAIN_PROVIDER; falls back to the template summary
        if the brain is unavailable so the dashboard never hard-fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app.app)


# --- speech store ----------------------------------------------------------


def test_store_and_get_speech_roundtrip():
    sid = app.store_speech("hello world")
    assert isinstance(sid, str) and len(sid) >= 16
    assert app.get_speech(sid) == "hello world"


def test_get_unknown_speech_returns_none():
    assert app.get_speech("does-not-exist") is None


def test_expired_speech_returns_none(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(app, "_now", lambda: t["now"])
    sid = app.store_speech("transient")
    t["now"] = 1000.0 + app.SPEECH_TTL_SECONDS + 1
    assert app.get_speech(sid) is None


# --- /api/jarvis/speak now requires speech_id (no raw text) -----------------


def test_speak_rejects_raw_text(client):
    resp = client.post("/api/jarvis/speak", json={"text": "make jarvis say anything"})
    assert resp.status_code == 400


def test_speak_unknown_speech_id_returns_404(client):
    resp = client.post("/api/jarvis/speak", json={"speech_id": "nope"})
    assert resp.status_code == 404


def test_speak_with_valid_speech_id_speaks_stored_text(client, monkeypatch):
    captured = {}

    def fake_synth(text):
        captured["text"] = text
        return b"RIFF0000WAVE", {}

    monkeypatch.setattr(app, "synthesize_jarvis_audio", fake_synth)
    sid = app.store_speech("the server owns this sentence")
    resp = client.post("/api/jarvis/speak", json={"speech_id": sid})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert captured["text"] == "the server owns this sentence"


# --- RAG synthesis + brain routing -----------------------------------------


def test_brain_synthesize_defaults_to_local_ollama(monkeypatch):
    monkeypatch.setattr(app, "call_ollama", lambda prompt: "OLLAMA_ANSWER")
    monkeypatch.setattr(app, "call_claude", lambda prompt: "CLAUDE_ANSWER")
    monkeypatch.setattr(app, "JARVIS_BRAIN_PROVIDER", "local")
    out = app.brain_synthesize("q", [{"collection": "c", "snippet": "s"}])
    assert out == "OLLAMA_ANSWER"


def test_brain_synthesize_claude_only_when_selected(monkeypatch):
    monkeypatch.setattr(app, "call_ollama", lambda prompt: "OLLAMA_ANSWER")
    monkeypatch.setattr(app, "call_claude", lambda prompt: "CLAUDE_ANSWER")
    monkeypatch.setattr(app, "JARVIS_BRAIN_PROVIDER", "claude")
    out = app.brain_synthesize("q", [{"collection": "c", "snippet": "s"}])
    assert out == "CLAUDE_ANSWER"


def test_rag_uses_brain_output(monkeypatch):
    monkeypatch.setattr(app, "brain_synthesize", lambda q, results: "Synthesized natural answer.")
    payload = {"results": [{"collection": "qoz", "snippet": "deadline Dec 31 2026"}],
               "total_hits": 1, "collections_searched": 1}
    out = app.synthesize_rag_answer("when is the QOZ deadline", payload)
    assert out["answer"] == "Synthesized natural answer."
    assert out["results"]  # display results still present


def test_rag_falls_back_to_template_when_brain_unavailable(monkeypatch):
    monkeypatch.setattr(app, "brain_synthesize", lambda q, results: None)
    payload = {"results": [{"collection": "qoz", "snippet": "deadline Dec 31 2026"}],
               "total_hits": 1, "collections_searched": 1}
    out = app.synthesize_rag_answer("when is the QOZ deadline", payload)
    # Template fallback still produces a speakable answer, never an exception.
    assert isinstance(out["answer"], str) and out["answer"]
    assert "qoz" in out["answer"].lower() or "searched" in out["answer"].lower()


def test_search_chat_returns_speech_id_for_stored_spoken_text(client, monkeypatch):
    monkeypatch.setattr(app, "search_qdrant_master",
                        lambda q: {"results": [{"collection": "c", "snippet": "snip"}],
                                   "total_hits": 1, "collections_searched": 1})
    monkeypatch.setattr(app, "brain_synthesize", lambda q, results: "Spoken answer here.")
    resp = client.post("/api/search-chat", json={"q": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert "speech_id" in body and body["speech_id"]
    # The stored speech is the server-synthesized spoken text.
    assert app.get_speech(body["speech_id"]) == "Spoken answer here."
