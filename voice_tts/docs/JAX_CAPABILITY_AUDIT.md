# JAX Capability Audit — Pre-Phase-1.5 Investigation

**Date:** 2026-05-18
**Author:** Claude via terminal.luxqozp.com browser session
**Status:** READ-ONLY investigation. Zero file edits to JAX, dashboard, or any service. No codex-ask. No API calls to Anthropic/Ollama. voice_tts and voicebox untouched.
**Gate:** Mike asked "look for answer using JAX and search PGX, then auto-escalate to Claude" — this doc determines whether JAX already does that and whether dashboard should call JAX instead of reimplementing RAG in Phase 1.5.

## 1. JAX current capabilities (what it actually is)

**Project location:** `/home/msr8109/jax-bot/` — single file `jax_bot.py` (20,500 bytes, Apr 28). NOT a container; runs as `jax-bot.service` systemd unit ("JAX Telegram Bot (Direct Python)").

**Bot token:** 8798832090 (Rule 27 protected — confirmed in code, do not touch in any future change).

**Authorization:** Hardcoded `ALLOWED_USER_ID = 7752072049`. Single-user Telegram bot.

### 1.1 Brain: Claude Code CLI subprocess (NOT Anthropic API, NOT Ollama)

`run_claude(prompt)` shells out to `/home/msr8109/.local/bin/claude` with flags:
- `-p <prompt>` — single-shot prompt mode
- `--dangerously-skip-permissions` — full tool access
- `--max-turns 20` — agent can iterate up to 20 turns
- `--output-format text`
- `cwd=/home/msr8109`, env HOME=/home/msr8109
- 120s timeout

This means **JAX's LLM brain is a FULL Claude Code agent with filesystem and bash access**, not a stateless API call. It can read files, run Qdrant queries, search L: drive — anything Claude Code can do.

### 1.2 Retrieval pipeline (JAX has its own RAG already)

JAX has a hybrid retrieval pipeline:
- **`_search_keyword()`** (line 183) — keyword via SEARCH_API
- **`_search_semantic()`** (line 214) — semantic via SEARCH_API
- **`_merge_hits()`** (line 261) — hybrid merge, top=5
- **`is_file_query()`** (line 111) — query classifier with `_DOC_KEYWORDS`, `_ENTITY_NAMES`, `_EXTENSIONS`, `_APEX_TERMS` (negative filter), year-pattern, length heuristic

Embed model: `all-MiniLM-L6-v2`. Direct Qdrant URL `http://localhost:6333` collection `ldrive_docs`. L: drive root `/mnt/bookkeeper/L/Desktop`.

### 1.3 Routing logic (lines 527-580, `handle_message`)

Exact order from incoming Telegram text:
1. `text == "ping"` → reply "pong" (health)
2. Followup commands ("read ", "open ", "summarize ", "send ") AND user has prior search → `handle_search_followup`
3. `is_file_query(text)` matches → `handle_file_search` (Qdrant + keyword hybrid)
4. **Default → `run_claude(text)`** — full Claude Code agent

**This is exactly the auto-escalation Mike described.** JAX already tries PGX search first when the query looks file-flavored, and falls through to Claude Code for everything else. The escalation is intent-based (classifier), not confidence-based (similarity score).

### 1.4 API surface

**NONE.** JAX is Telegram-only. Only two handlers:
- `CommandHandler("start", start_cmd, filters=allowed)`
- `MessageHandler(filters.TEXT & ~filters.COMMAND & allowed, handle_message)`

No HTTP server. No FastAPI, no aiohttp, no Flask in imports. The dashboard CANNOT call JAX over HTTP today.

### 1.5 Dependencies (from imports)

`telegram` (python-telegram-bot), `httpx` (used for SEARCH_API calls), `asyncio`, `pathlib`, `re`. No `qdrant-client` direct import — uses HTTP via SEARCH_API. No `anthropic`, no `ollama` SDK — uses Claude via subprocess.

## 2. Dashboard Search Chat current capabilities

**Project location:** `/home/msr8109/projects/cfo_pipeline/` (symlink to `/home/msr8109/repos/cfo_pipeline/`). Process: `python dashboard/dashboard.py` running from `.venv`.

### 2.1 Search backend

Dashboard reads `SEMANTIC_SEARCH_URL` env var, defaults to `http://localhost:8100` (variable named `SEMANTIC_SEARCH_BACKEND` in code, line 5733). This is **a DIFFERENT search service than JAX's** (JAX uses port 8097). Port 8100 exposes `/`, `/health`, `/collections`, `/search`.

### 2.2 Search UI

- Route `/semantic-search` (line 5736) — page render
- Route `/semantic-search/results` (line 5753) — htmx-driven results panel
- Calls `{SEMANTIC_SEARCH_BACKEND}/search?q=...&total=5`
- Renders raw hits; templated summary sentence sent to legacy voice service on port 8110 (not voice_tts)

### 2.3 No LLM synthesis

The dashboard search returns Qdrant hits + templated summary. No call to any LLM (Claude API, Ollama, or Claude Code) anywhere in the search code path.

## 3. Overlap and divergence

| Axis | JAX | Dashboard Search Chat |
|---|---|---|
| Retrieval backend | port 8097 (SEARCH_API) | port 8100 (SEMANTIC_SEARCH) |
| Qdrant collection focus | `ldrive_docs` | configurable, returns `collections_searched` |
| LLM synthesis | Claude Code CLI (full agent, 20-turn) | NONE |
| Query routing | Intent classifier (file-query vs general) | All queries treated identically |
| API surface | Telegram-only | HTTP routes available |
| Voice readback | None (text replies to Telegram) | Legacy port 8110 `/api/speak` |
| Voice cloning | N/A | Currently legacy, Phase 1 plans voice_tts migration |

**Two SEPARATE search backends** is the most surprising finding. Both wrap Qdrant; both probably duplicate logic. This is a refactoring opportunity but OUT OF SCOPE for Phase 1.5.

## 4. Phase 1.5 redesign options

### Option A — Original Phase 1.5 design (toggle, dashboard reimplements RAG)
Dashboard adds its own RAG pipeline: existing port-8100 search → LLM toggle (gemma3:27b default / Claude API opt-in) → voice readback. As written in `ADAPTER_DESIGN_PHASE1_5.md`.

**Pros:** Self-contained. No JAX dependency. Toggle gives explicit user control. Privacy-first default to local Ollama.

**Cons:** Third LLM brain on the box (gemma3:27b in addition to Claude Code via JAX and Claude API). User has to think about toggle. Doesn't match what Mike described ("auto-escalate"). Duplicates JAX's existing RAG logic. Force-local allowlist for sensitive collections is a parallel privacy mechanism JAX doesn't have.

### Option B — Dashboard calls JAX as the brain
Add a small HTTP endpoint to JAX (`POST /ask` taking `{text, mode}`) that runs the same routing JAX already does for Telegram. Dashboard search chat POSTs there. JAX returns the answer; dashboard speaks it via voice_tts.

**Pros:** Reuses JAX's existing intent classifier and Claude Code escalation. ONE LLM brain (Claude Code) instead of three. Matches Mike's stated intent. Dashboard becomes a thin client. Future surfaces (browser ext, Windows tray) all call the same endpoint.

**Cons:**
- Adds an HTTP server to JAX → small change to a Rule 27 service (token isolation matters). Must be local-only bind (127.0.0.1) with no auth-bypass risk.
- Couples dashboard uptime to JAX uptime (mitigation: fall back to templated search if JAX is down).
- JAX currently runs as single-user Telegram bot — its routing assumes one trusted caller. Multi-caller via HTTP needs the dashboard to identify itself (e.g., `X-Caller: dashboard-rag` and the JAX endpoint accepts it).
- Privacy posture: JAX uses Claude Code which uses Anthropic API. Mike's Phase 1.5 force-local allowlist (mike_files, financial collections etc.) does NOT exist in JAX today. **This is a real regression if dashboard sensitive queries route to Claude Code via JAX.**
- 120s timeout is long for an interactive dashboard search.

### Option C — Shared RAG library imported by both
Extract JAX's routing into a Python module (e.g., `~/projects/rag_core/`). JAX imports it. Dashboard imports it. Two callers, one logic.

**Pros:** Eliminates dual-implementation. Cleanest long-term.

**Cons:** Requires refactoring JAX (touching a Rule 27 service). Bigger lift than A or B. Coordinating venvs and deploys is annoying.

## 5. Recommendation

**Hybrid Option B-prime: dashboard calls JAX-as-brain for general queries, but Phase 1.5 force-local allowlist is enforced AT THE DASHBOARD LEVEL before calling JAX.**

Architecture:
1. Dashboard search chat receives query
2. Dashboard runs its current port-8100 retrieval to get top-K with collection metadata
3. Dashboard checks force-local allowlist: if ANY top-K hit comes from a sensitive collection (mike_files, ldrive_docs, bank_statements_2026, sales_reports, chatvault_chunks, ai_brain_unified, all `apex_*` collections, `self_learning_scores`) → **route to LOCAL gemma3:27b synthesis path** (still need to build this fallback for the sensitive case). Banner: "Routed locally for privacy."
4. Otherwise → POST to new JAX `/ask` endpoint with the question. JAX uses its existing routing (file_search → Claude Code).
5. Return text answer to dashboard. Dashboard sends to voice_tts. Speak.

**Why this:**
- Mike's privacy posture (uncrossable sensitive routing) is preserved
- JAX is reused for non-sensitive queries → ONE brain for those
- Force-local path still needs gemma3:27b, BUT only for sensitive queries → smaller surface, more justified
- Dashboard owns the privacy decision (collection allowlist), JAX owns the smart routing
- Failure modes: JAX down → fall back to templated search. Gemma3 down → fall back to templated search. Both must show banner.

**Required JAX change:** add a small FastAPI/aiohttp HTTP server bound to 127.0.0.1, no TLS (local), shared `JAX_RAG_API_KEY` env var, single endpoint `POST /ask`. Must NOT expose Telegram token, must NOT change `handle_message` or its three-way routing. Add `X-Caller: dashboard-rag` logging.

## 6. Risks per option

**Option A:**
- Three LLM brains to maintain
- Doesn't match Mike's auto-escalation request
- gemma3:27b quality on Mike's domain (finance, legal, business) is unproven

**Option B / B-prime:**
- Touching jax-bot.service (Rule 27 token isolation) for HTTP endpoint
- Dashboard now depends on JAX uptime
- Anthropic API egress for non-sensitive queries (acceptable per Phase 1.5 privacy posture)
- HTTP endpoint adds attack surface even bound to localhost
- 120s timeout on Claude Code subprocess is long for an interactive search

**Option C:**
- Largest refactor
- Two-codepath deploy coordination

## 7. Open questions for Mike

1. **Approve Option B-prime?** (Dashboard does retrieval + privacy decision; JAX-as-brain for non-sensitive; gemma3:27b for sensitive)
2. **Or stay with Option A (original Phase 1.5)?** (Toggle in dashboard, no JAX change)
3. **Or fold this into Phase 2** and ship Phase 1.5 as-is (toggle), then refactor later?
4. JAX `/ask` endpoint auth model: shared key, or IP-allowlist localhost only?
5. JAX `/ask` timeout: 120s same as Telegram, or shorter (e.g., 30s) for dashboard interactive use?
6. If Option B-prime: should the dual SEARCH_API services (8097 + 8100) be consolidated, or stay separate?
7. Sensitive allowlist source of truth: dashboard config (Phase 1.5 doc), JAX, or shared file?

## 8. Verification snapshot (what's actually running right now)

```
voice_tts        Up 2 hours (healthy)   127.0.0.1:8050->8050/tcp
voicebox         Up 4 hours (healthy)   127.0.0.1:17493->17493/tcp
jax-bot.service  loaded active running  JAX Telegram Bot (Direct Python)
port 6333        Qdrant — collections include apex_trade_theses, apex_seeking_alpha, apex_sentiment, apex_quant_math, self_learning_scores, apex_backtests, apex_knowledge, apex_reddit, apex_fred_data, and more
port 8097        SEARCH_API (FastAPI) — /health /search /read
port 8100        SEMANTIC_SEARCH (FastAPI) — / /health /collections /search
port 8110        Legacy Jarvis voice service (NOT FastAPI, used by dashboard search chat audio today)
```

No code or service was modified during this audit. Confirmed by checking process state and file mtimes (only `JAX_CAPABILITY_AUDIT.md` written, no other files in jax-bot/, dashboard/, voice_tts/ touched).
