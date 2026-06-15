# Round 4 — Audit P0/P1 Fix Decisions

Date: 2026-06-15
Branch: `fix/audit-p0-p1`
Consults: codex-ask (GPT-5.5), perplexity-ask (sonar-reasoning-pro), cursor-ask.
Raw outputs: `codex_consult.txt`, `perplexity_consult.txt`, `cursor_consult.txt`.

## Arm status
- **codex-ask** — usable verdict.
- **perplexity-ask** — usable verdict.
- **cursor-ask** — FAILED (`HTTP 401 authentication_error: invalid x-api-key`). Same
  broken-arm condition seen in prior rounds; not usable.

2 of 3 arms produced usable verdicts and they **converge on both questions**. Per
Rule 23 (2-of-3 confirm) the decisions below are accepted. No escalation to Mike:
both decisions *align with* documented architecture (Rule 36) rather than changing it.

---

## Decision Q1 — Brain routing for RAG synthesis (audit P1 #7)

**Decision: DEFAULT LOCAL.** RAG answer synthesis routes to a local Ollama model
(`gemma3:27b`) by default. Provider is selectable via env `JARVIS_BRAIN_PROVIDER`:
- `local` (default) → Ollama `JARVIS_OLLAMA_MODEL` (default `gemma3:27b`).
- `claude` → Claude API, explicit opt-in only.
- `template` → legacy templated summary (no LLM), used automatically as a
  graceful fallback when the selected brain is unreachable, so the dashboard
  never hard-fails.

No *automatic* cloud fallback. Claude is only ever called when the operator sets
`JARVIS_BRAIN_PROVIDER=claude`.

**Rationale:** Rule 36 (PGX-LOCAL-FIRST) directly governs this. An earlier codex
pass recommended defaulting to Claude *without knowing Rule 36*; both arms this
round explicitly reversed that once Rule 36 was in scope. Defaulting to Claude
would ship every search query + retrieved Qdrant snippets off-box, violating
Rule 36 and leaking internal context. gemma3:27b is strong enough for spoken
knowledge synthesis; this is not APEX trade-thesis generation (codex: "Do not
reuse APEX signal_synthesis→claude routing for Voice Box RAG").

## Decision Q2 — Close arbitrary-TTS-proxy hole (audit P0 #4)

**Decision: server-side speech store + `speech_id`** (both arms' option (a)).
- `/api/search-chat` synthesizes the answer server-side, stores
  `{speech_id, text, created_at}` with a short TTL, returns `speech_id` (+ display
  answer/results).
- `/api/jarvis/speak` accepts **`speech_id` only**; it looks up server-owned text
  and speaks it. Raw caller-supplied `text` is no longer accepted. Unknown/expired
  ids → 404.

**Rationale:** Only server-generated text becomes audio — a clean provenance
guarantee. Both arms rejected HMAC (still lets the client choose arbitrary text;
proves "a client asked," not "the server generated it") and reject/sanitize-raw
(brittle, easy to silently re-open). The greeting path is already server-built.

---

## Notes on already-landed items
- Audit P0 "TTS_API_KEY auth bypass via ENVIRONMENT": **already closed** in
  `server.py` by the §5.1 startup guard (no `ENVIRONMENT` conditional; requires
  `TTS_API_KEY` unless `ALLOW_UNAUTH_TTS=1`). Covered by `tests/test_startup_guard.py`.
- Audit P1 "X-Caller server-side logging": already landed (`tests/test_xcaller_logging.py`).
