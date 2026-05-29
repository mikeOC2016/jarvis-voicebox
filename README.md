# JARVIS Voice Box

Voice-enabled assistant layer for the MASTER DASHBOARD (PGX).

## Components

- **dashboard/** — MASTER DASHBOARD (`app.py`, FastAPI, port 8101). Hosts the Search Chat UI and the JARVIS voice integration (`synthesize_jarvis_audio`, `build_search_chat_answer`).
- **voice_tts/** — Voice TTS engine (`server.py`, Dockerized). Cloned-voice synthesis service consumed by the dashboard.

## Architecture (current)

Search Chat -> Qdrant semantic retrieval -> templated summary -> voice_tts -> spoken readback (cloned voice).

## Roadmap (planned)

Phase 1.5: Search Chat -> Qdrant retrieval -> LLM (brain toggle: local gemma3:27b / Claude API) -> cited answer -> voice_tts readback (RAG).

See voice_tts/docs/ for the adapter design docs.

## Review

This repo is staged for external code review (Greptile) plus internal Rule 23 triple-review (Cursor + Codex GPT-5.5).
