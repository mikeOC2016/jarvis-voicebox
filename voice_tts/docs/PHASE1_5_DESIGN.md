# Phase 1.5 — MASTER DASHBOARD Search Chat: RAG + LLM Synthesis + Voice

**Date:** 2026-05-19
**Status:** DRAFT — awaiting Rule 23 triple-review (cursor + codex)
**Predecessor:** PHASE1_5_RECON.md (this directory)
**Locked Phase 1a:** ADAPTER_DESIGN_PHASE1A.md (unchanged, still live)
**Scope architect:** Option C — shared `rag_core` library/service per Rule-23 convergent recommendation in Phase 1a doc lines 665, 688

---

## 1. Goal (one sentence)

Upgrade MASTER DASHBOARD Search Chat (`hub/app.py` port 8101) from template-only retrieval display to RAG + LLM synthesis with hardened voice readback, so the Search Chat box can answer arbitrary questions from PGX-indexed data and speak the answer.

## 2. Non-goals (locked OUT of Phase 1.5)

- **No touching CFO dashboard adapter** (Phase 1a, live). `cfo_pipeline/dashboard/dashboard.py` is not edited.
- **No touching JAX adapter** (Phase 1a, live). `~/jax-bot/jax_bot.py` is not edited.
- **No master dashboard event narration** (Phase 1b, deferred). No Sync-All, no status-change announcements added in 1.5.
- **No new Telegram bot, no Cloudflare exposure, no Windows tray, no browser extension** (Phase 2, deferred).
- **No retrieval changes to qdrant-master-search service.** Phase 1.5 consumes its existing API only.
- **No PAR POS path changes.** DQ sales special-case stays as-is in hub.

## 3. What changes

### 3.1 New shared library: `rag_core`
Location: `/home/msr8109/projects/rag_core/`
Python package with `pyproject.toml`, installed editable into hub's venv. Mirrors the `voice_text_utils` pattern.

**Public API (v1):**
```python
from rag_core import answer_query, AnswerResult

result: AnswerResult = answer_query(
    query: str,
    *,
    retrieval_url: str = "http://127.0.0.1:8100/search",
    llm_base_url: str = "http://127.0.0.1:11434",
    fast_model: str = "gemma3:27b",
    deep_model: str = "llama3.3:70b",
    timeout_s: float = 30.0,
    max_context_chars: int = 6000,
    allowlisted_collections: list[str] | None = None,
    caller: str = "master-dashboard-rag",
)
```

**`AnswerResult` dataclass:**
- `answer: str` — the displayed answer (HTML-safe plain text)
- `text: str` — the voice-readable trimmed version (≤220 chars via `voice_text_utils.trim_for_speech`)
- `results: list[dict]` — retrieved snippets (collection, score, snippet) for the source list under the chat answer
- `total_hits: int`
- `collections_searched: int`
- `mode: Literal["retrieval_template", "llm_synthesized", "llm_no_context"]` — audit which path was taken
- `model_used: str | None`
- `latency_ms: int`

### 3.2 Hub edits: `hub/app.py` only

**Edit 1:** Import `rag_core` and `voice_text_utils` at module top.

**Edit 2:** Replace the body of `api_search_chat` (line 384) with:
```python
if _is_dq_sales_today_query(query):
    answer = build_dq_sales_answer(query, fetch_dq_sales())
    return JSONResponse({"ok": True, "speakable": True, **answer})

result = rag_core.answer_query(query, allowlisted_collections=RAG_ALLOWLIST, caller="master-dashboard-rag")
return JSONResponse({
    "ok": True,
    "speakable": True,
    "answer": result.answer,
    "text": result.text,
    "results": result.results,
    "mode": result.mode,
})
```

**Edit 3:** Replace `synthesize_jarvis_audio(text)` body to route through voice_tts on 8050 with Bearer + X-Caller `master-dashboard-rag`, fall back to jarvis-voice on 8110 only when explicit fast-path needed (greeting only). This closes the security gap identified in recon §7.

**Edit 4:** Add `RAG_ALLOWLIST` env-driven config (default empty = all collections; can be restricted to e.g. {ldrive_docs, chatvault_chunks, sales_reports, apex_knowledge} for narrower retrieval surface).

**Edit 5:** Add `MASTER_DASHBOARD_TTS_API_KEY` env var to hub.service's EnvironmentFile (or shared `TTS_API_KEY` per locked-spec line 688). Fail-closed: if unset, `/api/jarvis/speak` returns HTTP 503 with logged `[TTS] caller=master-dashboard-rag skipped reason=missing_key`. Phase 1a pattern.

### 3.3 Files NOT touched (regression-protect)

- `cfo_pipeline/dashboard/dashboard.py` — Phase 1a CFO adapter live
- `~/jax-bot/jax_bot.py` — Phase 1a JAX adapter live
- `voice_tts/` source — already hardened in Phase 1a
- `voice_text_utils/` source — imported as-is, no modifications
- `jarvis_voice/` source — hub edit 3 changes the *caller*, not jarvis-voice itself (jarvis-voice hardening is Phase 1.6 if needed)
- `qdrant_master_search/` source — read-only consumer
- `master_file_lookup/` source — not used in Phase 1.5

## 4. RAG flow (the heart of Phase 1.5)

```
user types in Search Chat input
  → POST /api/search-chat {q: "..."}
  → hub.api_search_chat
      → if DQ-sales-today pattern: existing PAR POS template path (unchanged)
      → else: rag_core.answer_query(query)
          → STEP 1: search_qdrant_master(query) via 8100/search (k=6, total=12)
          → STEP 2: filter results by allowlisted_collections (if set)
          → STEP 3: branch on top score:
              │   top_score ≥ 0.75 AND retrieved snippet > 200 chars:
              │     → LLM_SYNTHESIZE mode: build prompt with top 4 snippets,
              │        call Ollama gemma3:27b, max_tokens=350, temperature=0.2,
              │        system: "Answer Mike's question using only the provided context. 
              │        Cite collection names in parentheses. If context is insufficient,
              │        say so plainly. Do not invent facts."
              │     → mode = "llm_synthesized"
              │   0.55 ≤ top_score < 0.75:
              │     → RETRIEVAL_TEMPLATE mode: current template path
              │        "I found N matches across M collections. Top result from {coll}: {snippet}"
              │     → mode = "retrieval_template"  
              │   top_score < 0.55 OR results == []:
              │     → LLM_NO_CONTEXT mode: call gemma3:27b WITHOUT retrieved context,
              │        system: "You are PGX assistant for Mike Ramadan. Answer briefly.
              │        If you don't know, say so." Hard truncate to 600 chars.
              │     → mode = "llm_no_context"
          → STEP 4: build AnswerResult, apply voice_text_utils.trim_for_speech(answer)
          → STEP 5: return
  → hub returns JSONResponse
  → browser JS displays answer + plays /api/jarvis/speak readback
```

Decision thresholds (0.75 / 0.55) are guesses based on the live test ("GCBW lease" → 0.732). They're config-driven via env so they can be tuned without code changes.

## 5. Security & isolation

### 5.1 Hub /api/jarvis/speak hardening (Edit 3 above)
- Adds Bearer auth via shared `TTS_API_KEY`
- Adds `X-Caller: master-dashboard-rag` header on outbound voice_tts calls
- Server-side enforces text ≤ 1200 chars (longer is truncated; spec aligns with 1024 in voice_tts hardening)
- Fail-closed: missing key → 503 + logged ERROR per Phase 1a pattern

### 5.2 Allowlist
Hub adds `master-dashboard-rag` to voice_tts allowlist (alongside existing `jax`, `cfo-dashboard-events`, `smoke`). Per locked-spec line 700, the allowlist values `["jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub"]` already accommodate this caller (with naming `dashboard-rag` vs `master-dashboard-rag` to be reconciled in Rule 23).

### 5.3 No PII / sensitive collection leakage
`RAG_ALLOWLIST` defaults to opt-in only safe collections. Sensitive collections (e.g. anything containing entity financials at row-level) require explicit inclusion. Rule 23 reviewer to confirm filter is server-side, not browser hint.

### 5.4 Cross-doc preservation
Phase 1a Adapter B (CFO) and Phase 1a JAX adapter both **continue to function unchanged**. Phase 1.5 only introduces new files (`rag_core/`) and edits hub/app.py + hub.service EnvironmentFile. No shared-helper changes (voice_text_utils stays at the version Phase 1a depends on).

## 6. Tests (gating implementation completion)

1. `rag_core` unit tests:
   - Mocked retrieval + mocked Ollama — verify all 3 branch decisions trigger correctly
   - Voice trim invoked, result.text ≤ 220 chars
   - LLM timeout falls back to retrieval template (don't crash search)
   - Allowlist filter applied post-retrieval, before LLM context build
   - Caller header propagation through voice path

2. Hub integration tests (no Ollama, no qdrant, all mocked):
   - DQ-sales path unchanged (regression)
   - Non-DQ path goes through rag_core
   - `/api/jarvis/speak` rejects with 503 when TTS_API_KEY missing
   - `/api/jarvis/speak` sends correct X-Caller

3. Phase 1a regression smoke (manual):
   - CFO dashboard sync-all still narrates (`cfo-dashboard-events` caller)
   - JAX `/voice` reply still works
   - voice_tts on 8050 still returns 401 without Bearer

4. Live retrieval smoke (post-implementation):
   - Query "GCBW lease term" → llm_synthesized mode, answer references lease agreement, voice trims to ≤220 chars
   - Query "random gibberish xyzzy" → llm_no_context mode, polite "I don't know"
   - Query "DQS sales today" → unchanged PAR POS template (regression)

## 7. Deferred to future phases (NOT in 1.5)

- **Phase 1.6:** jarvis-voice service hardening (Bearer + X-Caller parallel to voice_tts)
- **Phase 1b:** MASTER DASHBOARD event narration (Sync All, status changes speak)
- **Phase 1.7:** RAG citation rendering in the chat results list (clickable collection links)
- **Phase 2:** Cloudflare exposure, Windows tray, browser extension
- **Phase 3:** Streaming LLM responses (current spec is non-streaming, blocking)
- **Phase 4:** Cross-collection re-ranking / hybrid BM25+vector

## 8. Implementation order (when Mike green-lights)

1. Create `rag_core/` package skeleton + tests
2. Implement retrieval wrapper (calls qdrant-master-search)
3. Implement Ollama wrapper (gemma3:27b call, prompt building, timeout)
4. Implement branching logic + AnswerResult
5. Unit tests all green
6. Hub edits 1-5
7. Hub service env file + restart
8. Phase 1a regression smoke
9. Live retrieval smoke
10. Ping Mike, hand off

## 9. Rule-23 review request

This design doc requires:
- **Cursor SSH → PGX:** writer + architecture + silent-break pass. Confirm: imports/cron/env/services unchanged outside the listed edits. Confirm: no Phase 1a regression risk.
- **Codex-ask:** logic + edge cases pass. Confirm: branch thresholds reasonable, prompt template defensible, error handling exhaustive, allowlist enforcement correct.

Per Mike's Rule 23 update yesterday: Cursor REPLACES Perplexity. Two reviewers, both must confirm clean before implementation can begin.

## 10. Stop conditions (per Mike's tonight directive)

- If Cursor and Codex **disagree on architecture** (Option C vs. something else): STOP, ping Mike, wait.
- All other disagreements (typo fixes, wording, threshold tweaks): resolve, document, keep moving.

