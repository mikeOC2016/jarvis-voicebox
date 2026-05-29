# JARVIS Voice Box - Whole Codebase Audit Brief

## Goal
The voice box currently sounds like a cheap 1990s computer program. Audit the ENTIRE
codebase and recommend how to make it professional and properly adapted to the MASTER
DASHBOARD. AUDIT AND RECOMMEND ONLY - do NOT write or modify code in this pass.

## Components
- dashboard/app.py (750 lines) - MASTER DASHBOARD, FastAPI port 8101. Search Chat UI,
  /api/jarvis/greeting, /api/jarvis/greeting/speak, synthesize_jarvis_audio(),
  build_search_chat_answer(). Frontend plays audio blobs via playUrl().
- voice_tts/server.py (452 lines) - XTTS v2 voice-cloning TTS service. FastAPI:
  /health /voices /speak /speak/stream, API-key auth, LRU cache, text_normalize.

## Known symptoms
1. Spoken content is templated status strings ("I found N matches across X collections",
   "Generating Jarvis greeting") - robotic input produces robotic output.
2. Search Chat speaks a retrieval SUMMARY, not an LLM-synthesized answer (Phase 1.5 RAG
   gap). Sounds like a search box reading its own status bar.
3. Suspected polish gaps: no SSML/prosody pacing, possibly short/low-quality reference
   voice clip, abrupt audio start/stop, no assistant "persona" phrasing layer.

## What the engine already does well (verify, don't rip out)
XTTS v2 neural cloning, streaming, auth, caching, normalization. Foundation is sound.

## Audit scope (no limit)
Architecture, code quality, security, the dashboard<->voice_tts integration seam,
latency/perf, error handling, and SPECIFICALLY the path to natural-sounding speech:
RAG answer synthesis + a voice-persona layer. Recommend the default brain
(local gemma3:27b vs Claude API vs toggle) with your reasoning.

## Deliverable
Prioritized findings (P0/P1/P2), each with: file, problem, recommended direction.
A recommended implementation order. NO code in this pass.

## Existing design history (read these)
voice_tts/docs/ADAPTER_DESIGN_PHASE1*.md, PHASE1_5*.md, JAX_CAPABILITY_AUDIT.md,
CHATBOX_INVESTIGATION_2026_05_18.md, and prior verdicts in voice_tts/docs/rule23/.
