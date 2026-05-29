# Phase 1.5 — PGX Recon Findings

**Date:** 2026-05-19 (Tuesday evening)
**Author:** Claude (web) via Chrome MCP, autonomous recon
**Source data:** `~/projects/voice_tts/docs/PHASE1_5_RECON_RAW.txt` (44707 bytes raw)
**Status:** RECON COMPLETE — feeds design doc PHASE1_5_DESIGN.md

---

## 1. Critical correction to prior assumptions

The Phase 1a locked spec stated MASTER DASHBOARD lives at `hub/app.py` and Phase 1b targets port 8110. Recon shows:

- **Port 8101 = `hub.service` = MASTER DASHBOARD** (`WorkingDirectory=/home/msr8109/projects/hub`)
- **Port 8110 = `jarvis-voice.service` = Jarvis Voice Profile Service** (`WorkingDirectory=/home/msr8109/projects/jarvis_voice`)

These are **two separate uvicorn processes**, both named `app:app` but in different directories. `hub/app.py` (line 21) sets `JARVIS_VOICE_URL = "http://127.0.0.1:8110"` and calls it to synthesize audio. The Phase 1b reference to "hub/app.py on port 8110" in earlier recon notes was wrong — the master dashboard hub is 8101.

## 2. Service map (17 surfaces, all online)

| Port | Bind | Service | Working dir |
|---|---|---|---|
| 8050 | 127.0.0.1 | voice_tts (Phase 1a hardened) | (separate) |
| 8100 | 0.0.0.0 | qdrant-master-search | qdrant_master_search/ |
| 8101 | 0.0.0.0 | **hub.service (MASTER DASHBOARD)** | **hub/** |
| 8102 | 127.0.0.1 | master-file-lookup | master_file_lookup/ |
| 8110 | 0.0.0.0 | jarvis-voice.service | jarvis_voice/ |
| (others) | various | CFO dashboard, GCBW, APEX surfaces, etc. | various |

## 3. hub/app.py current state (401 lines)

### Existing endpoints
- `GET /api/status` — health probes
- `GET /api/jarvis/greeting` — returns greeting text JSON
- `POST /api/jarvis/greeting/speak` — synthesizes greeting WAV via `synthesize_jarvis_audio()`
- `POST /api/jarvis/speak` — **synthesizes arbitrary text up to 4096 chars to WAV. NO auth, NO allowlist, NO X-Caller header.** Bypasses Phase 1a's voice_tts hardening by going direct to jarvis-voice on 8110.
- `POST /api/search-chat` — current Search Chat router
- `GET /favicon.ico`

### Search Chat current logic (lines 383-405)
```python
if _is_dq_sales_today_query(query):
    answer = build_dq_sales_answer(query, fetch_dq_sales())
else:
    answer = build_search_chat_answer(query, search_qdrant_master(query))
return JSONResponse({"ok": True, "speakable": True, **answer})
```

- `_is_dq_sales_today_query()` = keyword match: ("dqs"|"dq"|"dairy queen") AND "sales" AND "today"
- `build_dq_sales_answer()` = pure template off PAR POS payload
- `build_search_chat_answer()` = pure template: "I searched PGX for {query}. I found {N} matches across {M} collections. Top result from {collection}: {snippet}" OR "did not find a strong match."
- **NO LLM in the loop.** No Ollama call, no gemma, no llama, no synthesis.

### Data sources reachable from hub
- `QDRANT_SEARCH_URL = http://127.0.0.1:8100/search` (qdrant-master-search)
- `DQ_SALES_URL = http://127.0.0.1:8090/api/dq/sales` (PAR POS bridge)
- `JARVIS_VOICE_URL = http://127.0.0.1:8110` (jarvis-voice TTS)

## 4. Live retrieval test

Query: `GCBW lease`
Result: 102 hits across 34 collections in <30ms. Top from `ldrive_docs` score 0.732: "This appears to be a lease agreement for the Gold Coast Beer Wine & Liquor store..."

Retrieval works. Synthesis does not.

## 5. RAG infrastructure inventory

### Ollama models on PGX (port 11434)
- Reasoning tier: `llama3.3:70b` (39GB), `qwen2.5:72b` (44GB), `gemma3:27b` (16GB)
- Fast tier: `gemma4:e4b` (8GB), `llama3.1:8b` (4GB)
- Coder: `qwen2.5-coder:32b`
- Embeddings: `nomic-embed-text` (used by qdrant_master_search backend)

### Qdrant collections (port 6333)
ldrive_docs, chatvault_chunks, sales_reports, apex_knowledge, apex_trade_theses, apex_macro_fred, apex_signals, apex_paper_trades, apex_dark_pool, apex_alt_economy, apex_ml_predictions, apex_perplexity, apex_reddit, apex_seeking_alpha, apex_quant_math, apex_sentiment, apex_fred_data, apex_google_trends, apex_backtests, self_learning_scores, apex_fred_data.

### Shared libs
- `voice_text_utils` at `/home/msr8109/projects/voice_text_utils/` — package installed, has `trim_for_speech` helper. Already used by Phase 1a (CFO adapter, JAX adapter).
- `rag_core` — **DOES NOT EXIST YET.** Phase 1.5 to create.

## 6. Z: vs local hub paths

- Local: `/home/msr8109/projects/hub/app.py` (canonical — `hub.service` WorkingDirectory)
- Z: `/home/msr8109/PGX-DATA/projects/hub/app.py` (Samba mirror, byte-identical, same mtime)

All Phase 1.5 edits target the LOCAL path. Z: is read-from-Windows mirror.

## 7. Security gap surfaced (Phase 1a regression candidate)

`POST /api/jarvis/speak` at hub line 363 accepts arbitrary text up to 4096 chars and calls jarvis-voice (8110) **without** Bearer, **without** X-Caller, **without** allowlist. Voice_tts on 8050 was hardened in Phase 1a, but hub's path is a different TTS service entirely (jarvis-voice on 8110, which apparently has no auth either).

**Phase 1.5 must close this** — either:
- (a) lock down jarvis-voice service to require Bearer + X-Caller (parallel hardening to voice_tts)
- (b) make hub's /api/jarvis/speak proxy through voice_tts on 8050 with the cfo-dashboard-events / new caller pattern

Recommend (b): consolidate on one hardened TTS endpoint per Phase 1a pattern. New X-Caller value: `master-dashboard-rag` (matches "dashboard-rag" in locked-spec allowlist line 700).

## 8. End of recon

Design doc next: `PHASE1_5_DESIGN.md`.
