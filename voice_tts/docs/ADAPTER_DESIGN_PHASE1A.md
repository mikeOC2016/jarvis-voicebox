# Voice TTS Adapter Design — Phase 1a (PGX-Local)

Date: 2026-05-18 (round 2 revision)
Author: Claude (revision of Phase 1 after Rule 23 NEEDS-CHANGES, then re-revised after Rule 23 round 2 NEEDS-CHANGES / CONDITIONAL)
Supersedes: `ADAPTER_DESIGN_PHASE1.md` (left in place as historical artifact)
Gate status: DESIGN ONLY. No implementation approved. No code, Docker, systemd, Cloudflare, JAX, or dashboard changes are included in this document. See §14 for the round-2 must-fix closure table.

## 0. Why this is a revision

Rule 23 (cursor + codex) returned NEEDS-CHANGES on the original Phase 1 + Phase 1.5 design pair. Both reviewers caught a critical scope error: Phase 1.5 targeted the wrong codebase (CFO Pipeline `dashboard.py` instead of MASTER DASHBOARD `hub/app.py`). They also surfaced shared must-fix items (truncation helper, TTS_API_KEY enforcement, failure cascade, tests) that crossed both phases.

Mike's response was to split Phase 1 into Phase 1a (JAX + CFO event narration only) and Phase 1b (MASTER DASHBOARD event narration in `hub/app.py`), and to rewrite Phase 1.5 against the correct codebase using Option C (shared RAG library).

This document is Phase 1a. It restricts the dashboard event narration adapter to the CFO Pipeline dashboard at `/home/msr8109/projects/cfo_pipeline/dashboard/dashboard.py` and removes any MASTER DASHBOARD reference. MASTER DASHBOARD event narration moves to `ADAPTER_DESIGN_PHASE1B.md`. MASTER DASHBOARD Search Chat / RAG moves to the rewritten `ADAPTER_DESIGN_PHASE1_5.md`.

Rule 23 round 1 must-fix items addressed in this doc:
- #1 Keep Phase 1 dashboard event narration isolated to approved CFO event paths.
- #5 Deterministic speech truncation helper, shared between JAX + CFO + future hub.
- #6 Enforce `TTS_API_KEY` on `voice_tts` in production; reject startup if unset.
- #9 Failure cascade: TTS down → text-only / visual-only, never escalate quietly.
- #11 Length-cap edge cases via deterministic split + word-boundary fallback.
- #12 Tests covering no key exposure, length caps, citation stripping (where relevant), TTS failure fallback.

Rule 23 round 1 items deferred or owned by other phases:
- #2, #3, #4, #7 → Phase 1.5 rewrite (Option C / `rag_core`).
- #8 → JAX `/ask` is NOT in Phase 1a; if it ever ships, it lives in Phase 1.5 only after Option C is in place.
- #10 → cache observability is Phase 2.

Rule 23 round 2 must-fix items addressed in this revision (full closure table in §14, source: `docs/rule23/round2/SUMMARY.md`):
- R2-1 `TTS_API_KEY` auth bypass risk — no more `ENVIRONMENT`-based bypass; key required by default; explicit `ALLOW_UNAUTH_TTS=1` only for dev/test (§5.1).
- R2-2 `/api/events/speak` raw-text vulnerability — caller-supplied text rejected; server constructs all speech from structured fields (§3).
- R2-3 HIGH leak narration removed from Phase 1a entirely; deferred to Phase 1a.5 with its own design pass (§3, §9.1, §10).
- R2-4 Adapters fail-closed if their local TTS key env var is empty (§2.2, §3.2).
- R2-5 `voice_tts` service-side `X-Caller` sanitization + logging (§1, §5.2).
- R2-6 `voice_text_utils` test list rewritten to remove internal contradiction (§4.4).
- R2-7 CFO event allowlist held at exactly `["sync_all_summary"]`; loosening requires its own Rule 23 pass (§3.2).

## 1. Service contract recap

`voice_tts` runs on PGX only at `127.0.0.1:8050`, backed by XTTS-v2 with one canonical reference voice `mike`. The contract was verified read-only from `/home/msr8109/projects/voice_tts/server.py` and from a live `GET /health` response.

Live `GET /health`:

```json
{
  "status": "ok",
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "model_loaded": true,
  "gpu_available": true,
  "device": "cuda",
  "sample_rate": 24000,
  "voices": ["mike"],
  "speaker_latents_cached": ["mike"],
  "audio_cache_size": 4
}
```

Endpoints:

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| `GET` | `/health` | none | none | JSON liveness/model state. Returns `status: "ok"` after model load; otherwise `"loading"`. |
| `GET` | `/voices` | bearer | none | `{"voices": ["mike"]}`. |
| `POST` | `/speak` | bearer | JSON `{"text": str, "voice": str = "mike", "language": str = "en"}`; `text` min 1 max 2000 chars | `audio/wav` body, 24 kHz mono PCM WAV. Headers: `X-Synthesis-Ms`, `X-Voice`, `X-Language`, `X-Sample-Rate`, `X-Audio-Bytes`, `Cache-Control: no-store`. |
| `POST` | `/speak/stream` | bearer | same schema as `/speak` | `audio/wav` `StreamingResponse` with WAV header and PCM chunks. Headers: `X-Voice`, `X-Language`, `X-Sample-Rate`, `Cache-Control: no-store`, `X-Accel-Buffering: no`, `CF-Cache-Status: BYPASS`. |

Auth contract (verbatim from `server.py:111` and request handlers):

- All endpoints except `/health` use `Authorization: Bearer <TTS_API_KEY>`.
- If `TTS_API_KEY` is unset in the service process, auth is currently bypassed. **Phase 1a treats this as a production bug to close — see §5.1. The new rule removes any `ENVIRONMENT`-based bypass; the only escape hatch is the explicit `ALLOW_UNAUTH_TTS=1` dev/test flag.**
- Missing bearer token returns `401` with detail `missing bearer token`.
- Wrong bearer token returns `403` with detail `invalid token`.

`voice_tts` service-side `X-Caller` handling (NEW in Phase 1a — required service change, see §5.2):

- Every `/speak` and `/speak/stream` request handler reads the `X-Caller` request header.
- Header value is sanitized server-side against the fixed allowlist: `["jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub"]`.
- Any value outside the allowlist (including empty/missing) is normalized to `"unknown"` for logging only. The request is NOT rejected on `X-Caller` alone; sanitization is for log hygiene, not auth.
- Each request emits exactly one structured log line: `[TTS] caller=<sanitized X-Caller> status=<200|4xx|5xx> latency_ms=<n>`.
- The spoken text body and the bearer token MUST NEVER appear in log output. No partial-text echo, no token prefix logging.
- `X-Caller` is informational metadata only. Auth remains bearer-token-based.

Request validation and failure modes:

- `/speak` and `/speak/stream` use Pydantic `SpeakRequest` with `text` length `1..2000`, default `voice="mike"`, default `language="en"`.
- Unknown voice raises `HTTPException(404, "voice not found: <voice>")`.
- Model missing/not loaded raises `503 "model not loaded"`.

Cache semantics:

- `server.py` normalizes text with `normalize_tts_text(text)` before hashing.
- Cache key is `xxh3_64(normalized_text + NUL + voice + NUL + language)`.
- Two requests cache-hit only when normalized text, voice, and language match exactly.
- `/speak` uses the LRU audio cache. `/speak/stream` does not currently populate or read it.
- LRU capacity is `LRU_CAPACITY` env var, default `1000`. Eviction is oldest-entry first via `OrderedDict.popitem(last=False)`.

Latency profile:

- Warm cache: `X-Synthesis-Ms: 0`, single-digit ms total elapsed.
- Cold/new short phrase: ~500-1200 ms.
- Cold number-heavy phrase: ~4 s because text normalization expands numbers.
- `/speak/stream` is faster first-byte but not used by Phase 1a adapters.

Service env (from `.env.example` and `server.py`):

| Env var | Meaning |
|---|---|
| `TTS_API_KEY` | Bearer token expected by `/voices`, `/speak`, `/speak/stream`. |
| `DEFAULT_VOICE` | Default voice if request omits it; stays `mike`. |
| `DEFAULT_LANG` | Default language if request omits it; stays `en`. |
| `LRU_CAPACITY` | Max complete clips in in-memory LRU; default 1000. |
| `LOG_LEVEL` | Python logging level. |
| `WARMUP_TEXT` | Startup warmup phrase. |
| `VOICES_DIR` | Reference voice mount path, default `/voices`. |
| `CACHE_DIR` | Model cache mount path, default `/cache`. |
| `XTTS_MODEL` | Model name, default XTTS-v2. |

## 2. Adapter A — JAX — Telegram voice notes

This entire section is unchanged from the original Phase 1 design except where explicitly noted in §4 (shared truncation helper) and §5 (TTS_API_KEY enforcement). Rule 23 did not call out JAX-specific issues; the must-fix items applied to shared concerns.

### 2.1 Current state

JAX root: `/home/msr8109/jax-bot`.

Entry point: `/home/msr8109/jax-bot/jax_bot.py`, launched as `/usr/bin/python3 /home/msr8109/jax-bot/jax_bot.py`. Confirmed via `jax-bot.service` systemd unit.

Line count: 600 lines (confirmed via `wc -l`).

Framework:

- `python-telegram-bot` 22.7.
- Uses `telegram.ext.Application`, `CommandHandler`, `MessageHandler`, `filters`, and `ContextTypes`.
- Polling mode: `app.run_polling(drop_pending_updates=True, allowed_updates=["message"])`.
- Existing HTTP client: `httpx` 0.28.1 is already installed via `python-telegram-bot`.

Token isolation (Rule 27):

- `BOT_TOKEN` with token prefix `8798832090` is hardcoded in `jax_bot.py`.
- This design does not change, move, log, rotate, or otherwise touch that token.
- Any implementation must preserve Rule 27 token isolation. No full-token logging.

Config pattern:

- No `requirements.txt`, `pyproject.toml`, `.env`, or external config file found in `/home/msr8109/jax-bot`.
- Module-level constants in `jax_bot.py`: `BOT_TOKEN`, `ALLOWED_USER_ID`, `CLAUDE_BIN`, `WORKDIR`, `CLAUDE_TIMEOUT`, `TG_LIMIT`, search endpoints, and L-drive paths.
- For TTS, add env-var reads with safe defaults; do not introduce a config system in Phase 1a.

Current response pipeline:

1. `handle_message(update, context)` receives text messages from the allowed Telegram user.
2. Logs incoming text to `/home/msr8109/jax-bot/jax.log`.
3. Sends `ChatAction.TYPING`.
4. Routes:
   - `ping` → direct `msg.reply_text("pong")`.
   - `read/open/summarize/send N` follow-ups → `handle_search_followup()`.
   - file-search intents → `handle_file_search()`.
   - all other text → `run_claude(text)` using `claude -p`, then reply chunks.
5. Long text is split by `chunk_text()` to stay under Telegram text limits.
6. Replies use `msg.reply_text(...)`; files use `context.bot.send_document(...)`.

Telegram send call sites (Phase 1 design audit lines, preserved):

- `handle_file_search`: line 319 `await msg.reply_text(piece)`.
- `_send_file`: lines 342, 348, 351, 356, 362, 385, 387 `reply_text`; lines 368-376 `context.bot.send_document(...)`.
- `handle_search_followup`: lines 397, 402, 411, 422, 431, 458, 467, 471 `reply_text`.
- `keep_typing`: line 480 `bot.send_chat_action(...)`.
- `start_cmd`: line 519 `update.message.reply_text(...)`.
- `handle_message`: line 542 `send_chat_action`, line 545 ping `reply_text`, lines 578-579 Claude reply chunks `reply_text`.
- `ignore_unauth` only logs.

The best adapter seam is a small helper called only after final user-visible text is produced. Start with the Claude reply path at `handle_message` lines 567-580 and optionally direct `ping` later. File-search lists and document summaries are noisy and stay opt-in.

### 2.2 Proposed integration

Mike decisions locked for Phase 1a (unchanged from Phase 1):

- Opt-in only.
- Triggers: `/voice <prompt>` command or `voice: <prompt>` prefix.
- Default JAX behavior remains text-only.
- WAV audio attachment first; OGG/Opus native Telegram voice notes are Phase 2.
- Scope: normal conversational replies only when `/voice` or `voice:` is invoked.
- No automatic speaking of file-search results or document summaries.
- Clarification: if Mike invokes `/voice` on a search query, JAX may speak whatever reply that path returns; auto-voice on search remains NO.

Data flow:

```text
Telegram message
  -> JAX detects /voice or voice: prefix
  -> JAX strips trigger and routes remaining prompt through the existing pipeline
  -> JAX sends the full text reply normally
  -> JAX chooses TTS text via shared trim_for_speech() helper (see §4)
       max 300 chars for jax caller
  -> POST http://127.0.0.1:8050/speak
       Authorization: Bearer <TTS_API_KEY>
       X-Caller: jax
       JSON {"text": tts_text, "voice": "mike", "language": "en"}
  -> receive WAV bytes
  -> send WAV audio attachment to Telegram
```

Telegram message format:

- Phase 1a uses WAV audio attachment (`send_audio` preferred if Telegram accepts the file object cleanly; `send_document` fallback if needed).
- OGG/Opus voice-note bubbles are Phase 2 (requires ffmpeg/Opus path).

Auth and attribution:

- JAX uses the shared `TTS_API_KEY` for Phase 1a.
- JAX sends `X-Caller: jax` on every TTS request.
- Per-adapter keys are Phase 2.
- The Telegram bot token (prefix `8798832090`) is not moved, changed, printed, rotated, or involved in the TTS auth path.

Hard precondition — adapter fail-closed on missing TTS key (Rule 23 round 2 #4):

- Before sending any request to `/speak`, JAX checks the local `JAX_TTS_API_KEY` (or shared `TTS_API_KEY`) env var.
- If the key is empty/unset, JAX MUST log `[TTS] caller=jax skipped reason=missing_key` at ERROR level once per cold start and at WARNING level (rate-limited) on subsequent attempts.
- JAX MUST NOT call `/speak` without an `Authorization: Bearer ...` header. The adapter is fail-closed, not fail-open. Sending an unauthenticated `/speak` is forbidden even during local development.
- The normal text reply path is untouched; the user still receives the text. Only the voice attachment is skipped.

Recommended JAX env knobs after approval:

- `JAX_TTS_MODE=off|command`, default `command` after rollout, `off` for emergency kill switch.
- `JAX_TTS_URL=http://127.0.0.1:8050`.
- `JAX_TTS_API_KEY=<shared service key>`.
- `JAX_TTS_VOICE=mike`.
- `JAX_TTS_LANG=en`.
- `JAX_TTS_MAX_CHARS=300`.

Failure mode (Rule 23 #9):

- Text-first always. The Telegram text reply must land before any TTS attempt and must not depend on TTS success.
- If `/speak` returns non-200, times out, or raises, JAX logs `[TTS] failed caller=jax status=... err=...` and continues text-only. No retry. No fallback to a different TTS service. No escalation to a cloud LLM. The user gets the normal text reply.
- TTS failure must never suppress or delay the normal text reply beyond the explicit voice attachment attempt.
- User-visible TTS failure messages stay off by default unless Mike asks for them.

Length cap (Rule 23 #5, #11):

- `voice_tts` accepts up to 2000 chars; Phase 1a JAX cap is 300 chars for spoken text.
- JAX always invokes `trim_for_speech(text, cap=300)` from the shared helper described in §4.
- If the final JAX text reply is over 300 chars, JAX still sends the full text in Telegram and speaks only the trimmed result.
- The helper guarantees a deterministic hard cap; JAX never sends raw text to `/speak`.

Opt-out / kill switch:

- `JAX_TTS_MODE=off` disables all voice without code changes.
- `JAX_TTS_MODE=command` enables `/voice` and `voice:` triggers only.
- No persistent per-chat state in Phase 1a.

Latency decision:

- Mike accepts 500-1200 ms for new short JAX phrases.
- No streaming in Phase 1a.

### 2.3 Files JAX would touch

No files are changed in this design pass. If Mike approves implementation after Rule 23, likely changes:

- `/home/msr8109/jax-bot/jax_bot.py`
  - Add env-var config constants for TTS.
  - Add trigger parsing for `/voice <prompt>` and `voice: <prompt>`.
  - Add `async synthesize_voice(text: str) -> bytes | None` using existing `httpx.AsyncClient`.
  - Add a delivery helper `send_tts_audio(update, context, text)`.
  - Import shared `trim_for_speech` from the new `voice_text_utils` module (see §4).
  - Call the helper only after the normal text reply on explicitly voiced paths.
  - Add loud log lines for skipped/truncated/failed/succeeded TTS calls.
- `/home/msr8109/jax-bot/test_file_search.py` or a new test file
  - Pure tests for trigger parsing, search-query opt-in behavior, and length-cap behavior without hitting Telegram or TTS.
- JAX process environment/startup wrapper
  - Add `JAX_TTS_*` env vars after locating the actual launcher (likely the systemd unit `jax-bot.service`).

### 2.4 New dependencies

For WAV audio attachment:

- No new Python dependency. JAX already has `httpx` and `python-telegram-bot`.
- Uses standard library `io`/`tempfile` if needed.

For native Telegram voice notes (Phase 2):

- Requires system `ffmpeg` with Opus support, or a Python audio encoding package.
- Not added in Phase 1a.

### 2.5 Risks

- Token isolation: JAX token prefix `8798832090` must not be changed, printed, rotated, or moved in this adapter pass.
- Telegram file limits: Bot API upload limit is commonly 50 MB; short WAVs are hundreds of KB.
- Telegram UX: WAV audio attachment is less polished than voice note; voice note requires Phase 2.
- Audio cache memory: Repeated common phrases cache well; unique long JAX replies churn the shared LRU.
- Temp files: If Telegram API requires file-like objects, implementation must clean temp files even on exceptions.
- TTS service outage: Degrade to text-only, loudly logged (see §6 failure cascade).

### 2.6 Verification plan

Read-only design; no verification executed yet. Proposed verification:

1. Unit test trigger parsing:
   - `voice: hello` → voice requested, prompt `hello`.
   - `/voice hello` → voice requested, prompt `hello`.
   - normal text with `JAX_TTS_MODE=command` → no voice.
2. Unit test `trim_for_speech(reply, cap=300)` via the shared helper (§4):
   - reply ≤ 300 chars → full reply.
   - reply > 300 chars with two sentences ending in `.` → first two sentences.
   - reply > 300 chars with no sentence punctuation → word-boundary trim at 300.
   - empty reply → empty string; caller skips TTS.
3. Dry-run TTS helper test with a fake HTTP client:
   - 200 with WAV bytes → returns bytes and headers.
   - 401/403/404/503 → returns `None`, logs error.
   - timeout → returns `None`, logs error.
4. Manual Telegram test in a controlled JAX session:
   - Send `/voice JAX bot ready. Standing by.`
   - Expect normal text reply first.
   - Expect WAV audio attachment second.
5. Failure test:
   - Temporarily point `JAX_TTS_URL` at an unused local port in a test process only.
   - Send `/voice test`.
   - Expect text reply; no crash; `[TTS] failed` in JAX log.
6. Regression:
   - Existing `ping`, file search, `read N`, `open N`, `summarize N`, `send N` still behave text/file-only unless explicitly voiced.

## 3. Adapter B — CFO Dashboard event narration (CFO Pipeline only)

This adapter is restricted to `cfo_pipeline/dashboard/dashboard.py`. MASTER DASHBOARD (`hub/app.py`) event narration is owned by Phase 1b. Search Chat / RAG is owned by Phase 1.5.

**Phase 1a scope after Rule 23 round 2 (R2-3 / R2-7):** CFO event narration is the **sync-all final summary only**. HIGH leak narration is removed from Phase 1a entirely and deferred to Phase 1a.5 (see §9.1) because it requires a real-time browser delivery channel (SSE / WebSocket / polling) that is not in Phase 1a scope. The CFO event allowlist for Phase 1a is exactly `["sync_all_summary"]`. No HIGH leak. No MED/LOW. No individual Plaid/QBO/QBWC pings. Loosening this allowlist requires its own Rule 23 pass.

### 3.1 Current state — confirmed by reading dashboard.py

Dashboard root: `/home/msr8109/projects/cfo_pipeline/` (symlinked from `/home/msr8109/repos/cfo_pipeline/`).

Entry point and serving:

- Active process: `/home/msr8109/projects/cfo_pipeline/.venv/bin/python dashboard/dashboard.py`.
- Framework: FastAPI with Uvicorn.
- Direct run: `uvicorn.run(app, host="0.0.0.0", port=8098)`.
- Browser-facing public route is `cfo.luxqozp.com`. Phase 1a does not alter Cloudflare.

Line count: 5793 lines (confirmed via `wc -l`).

Existing dashboard event/notification surfaces (verified via grep):

- Plaid Link status messages use inline JS `addStatus(msg, type)` (defined at line 2839) and an auto-removing DOM node near the Link Accounts button.
  - Sample messages from grep:
    - Line 2871: `'Plaid link token failed: ' + tokenResp.status + ' ' + errText`
    - Line 2905: `Re-link complete for ${inst}. Accounts and balances refreshed.`
    - Line 2912: `inst + ': ' + added + ' added, ' + updated + ' updated, ' + skipped + ' duplicates skipped.'`
    - Line 2917: `'Exchange failed: ' + r.status + ' ' + t`
    - Line 2921: `'Exchange error: ' + e.message`
    - Line 2936: `'openPlaidLink error: ' + e.message`
- Institution tiles use `.tile-sync-msg` (CSS line 1633) for sync progress and result messages. Per-tile message element queried at line 3089.
- Sync-all button is wired in `setupSyncAll()` at lines 3196-3225:
  - Element id `sync-all-btn` (HTML at line 3296).
  - Click handler iterates `tile.dataset.itemIds`, posts `/api/items/{id}/sync?balances_only=1` per item, updates button text per item.
  - Completion is detected client-side at lines 3217-3218 based on `ok === allItemIds.length`.
  - This is the canonical Phase 1a sync-all summary event.
- Leak events are persisted in `cfo.leak_events` and rendered under `/events` and `/events/{event_id}`. Confirmed SQL at:
  - Line 5077: `SELECT id, detected_at, event_type, severity, merchant_norm, annualized_impact, message, acknowledged, stream_id FROM cfo.leak_events`
  - Line 5119: `SELECT id, detected_at, event_type, severity, stream_id, transaction_id, merchant_norm, annualized_impact, message, alerted, acknowledged FROM cfo.leak_events`
  - Line 5090: severity rendered with class via `pct_class(row['severity'])`.
  - Line 5191: detail page `<b>{event['severity']}</b>`.
  - Lines 5226, 5238: `UPDATE cfo.leak_events SET acknowledged=...`.
  - Event types observed in code: `DUPLICATE`, `PRICE_HIKE`, `ZOMBIE`, `ENTITY_MISALLOC`, `WEBHOOK`.
- `scripts/daily_sync.py` (referenced from Phase 1) sends Telegram text alerts for HIGH/MED severity leak events through `send_telegram()` and marks `alerted=TRUE`.
- There is no active SSE, WebSocket, or continuous browser polling in `dashboard/dashboard.py`. Browser interactions are fetch-on-click.

Current Voicebox/TTS integration in CFO dashboard:

- Active CFO dashboard search (excluding `.venv` and backups) found NO `voicebox`, `VOICEBOX`, `17493`, `voice_tts`, `127.0.0.1:8050`, `/speak`, `send_voice`, or `send_audio` call sites.
- Voicebox container runs separately at `127.0.0.1:17493`, but CFO dashboard has no active call site.
- Therefore Phase 1a CFO adapter is a NEW event narration surface, not a replacement of existing dashboard Voicebox code.

Existing dependencies/config:

- `dashboard/dashboard.py` already imports `requests` and `httpx as _httpx_pi`.
- `.venv` has `httpx`, `requests`, FastAPI, Uvicorn, psycopg2, and Plaid installed.
- Config pattern is direct `os.environ[...]` / `os.environ.get(...)` at module import time.

### 3.2 Proposed integration

Mike decisions locked for Phase 1a (revised after Rule 23 round 2):

- CFO dashboard speaks the **sync-all final summary only**. HIGH leak narration is removed from Phase 1a entirely (deferred to Phase 1a.5; see §9.1).
- MED/LOW leak events remain visual only.
- Individual Plaid, QBO, and QBWC pings remain visual only.
- The CFO event allowlist is exactly `["sync_all_summary"]`. Loosening it requires its own Rule 23 pass.
- CFO dashboard remembers audio-enabled state in `localStorage` after first unlock.
- A visible mute/unmute toggle is always present.
- Voicebox stays running as failover during Phase 1a (retirement is separate Rule 23 after two weeks of stable adapter usage).

Phase 1a trigger policy:

1. Sync-all final summary (the ONLY narrated event in Phase 1a)
   - Fired exactly once when the `setupSyncAll()` loop at line 3196-3225 finishes.
   - The browser writes the structured per-tile result map (success/failure counts, optional list of failed institution slugs) into `sessionStorage.cfoLastSyncSummary` immediately before `window.location.reload()`.
   - On reload, the page reads `sessionStorage.cfoLastSyncSummary`, clears it atomically, and — only if audio is unlocked — calls `POST /api/events/speak` with `{"event": "sync_all_summary", "ok": <int>, "total": <int>, "failed_items": [<slug>, ...]}` (NO `text` field).
   - The dashboard backend constructs the canonical speech string from those fields using a hard-coded template (see schema below).
   - Speak one summary only. Do not speak per-item progress.

Server-derived canonical templates (R2-2 closure — backend constructs all speech; caller-supplied text is rejected):

| Branch | Condition | Canonical string |
|---|---|---|
| All OK | `ok == total` | `CFO dashboard online. Plaid sync complete.` |
| Partial / all-failed | `ok < total` | `Plaid sync failed for <N> item(s). Review the dashboard.` where `<N> = total - ok` |

Request schema for `POST /api/events/speak`:

```json
{
  "event": "sync_all_summary",
  "ok": 7,
  "total": 7,
  "failed_items": []
}
```

Rules:

- `event` MUST equal one of the allowlist values (currently only `"sync_all_summary"`).
- `ok` and `total` MUST be non-negative integers with `ok <= total`.
- `failed_items` is optional and informational only; the canonical template uses count, not item names. Future tightening may incorporate up to N item names but only via a template the server controls.
- ANY `text`, `narration`, `message`, or free-form string field in the request body is REJECTED with HTTP 400. The backend never echoes a caller-supplied string to `/speak`.
- The endpoint is authenticated by the dashboard's existing same-origin cookie/session model. It does not accept the TTS bearer token from the browser.

Data flow (sync-all only):

```text
Browser sync-all click finishes
  -> JS writes sessionStorage.cfoLastSyncSummary = {ok, total, failed_items}
  -> window.location.reload()
  -> on page load, JS reads + atomically clears sessionStorage.cfoLastSyncSummary
  -> if audio unlocked, JS POSTs structured fields (NO text) to /api/events/speak
  -> dashboard backend validates event ∈ allowlist and field types
  -> dashboard backend constructs canonical speech via hard-coded template
  -> dashboard backend invokes trim_for_speech(canonical, cap=220)
  -> dashboard backend POSTs http://127.0.0.1:8050/speak
       Authorization: Bearer <TTS_API_KEY>
       X-Caller: cfo-dashboard-events
       JSON {"text": canonical, "voice": "mike", "language": "en"}
  -> dashboard backend proxies WAV to browser as same-origin audio
  -> browser plays the WAV (audio already unlocked by the time it makes this request)
```

Hard precondition — adapter fail-closed on missing TTS key (Rule 23 round 2 #4):

- Before sending any request to `/speak`, the CFO dashboard backend checks the configured TTS key env var (`DASHBOARD_TTS_API_KEY` or shared `TTS_API_KEY`).
- If the key is empty/unset, the backend MUST log `[TTS] caller=cfo-dashboard-events skipped reason=missing_key` at ERROR level once per process start and at WARNING level (rate-limited) on subsequent attempts. The backend returns HTTP 503 to the browser and the visual feed continues unaffected.
- The backend MUST NOT call `/speak` without an `Authorization: Bearer ...` header. The adapter is fail-closed, not fail-open. Sending an unauthenticated `/speak` is forbidden even during local development.
- A missing key blocks ONLY the voice path. The sync-all visual outcome (tile messages, button state, page reload) is unaffected.

Browser autoplay constraint:

- Chrome blocks autoplay until the page has a user gesture.
- Dashboard uses an unlock pattern:
  - Always show a visible speaker/mute toggle in the existing `link-cluster` div near line 3296 (alongside `sync-all-btn` and `link-accounts-btn`). No new topbar surface in Phase 1a.
  - First click unlocks audio and stores enabled state in `localStorage.cfoEventsAudioEnabled = "1"`.
  - After first unlock, the dashboard remembers enabled state across sessions; the visible mute toggle remains available.
  - Until unlocked/enabled, dashboard keeps visual toasts and `tile-sync-msg` only and does not treat blocked audio as an error.
- No attempt to bypass browser autoplay policy.

Audio delivery design:

- Browser must not call `voice_tts` directly. `voice_tts` is bound to `127.0.0.1` on PGX; the browser cannot reach it from Mike's Windows machine through `cfo.luxqozp.com` without a proxy.
- Browser must never see `TTS_API_KEY`.
- Backend `POST /api/events/speak` is the only ingress from browser to TTS:
  - Backend validates `event` against the `["sync_all_summary"]` allowlist and field types/ranges.
  - Backend constructs the canonical speech string server-side; caller text is never used.
  - Backend enforces length cap via the shared `trim_for_speech` helper.
  - Backend calls `http://127.0.0.1:8050/speak` with `Authorization: Bearer <TTS_API_KEY>` and `X-Caller: cfo-dashboard-events`.
  - Backend returns `audio/wav` to browser with `Cache-Control: no-store`.
- Sync-all uses canonical fixed phrases (one of two branches) — high cache-hit rate on the voice_tts LRU.

Voicebox migration:

- Active CFO dashboard call sites to migrate: none found.
- Keep Voicebox container running as failover during Phase 1a.
- Do not retire or stop Voicebox in Phase 1a.
- Voicebox retirement is a separate Rule 23 decision (combined with Phase 1b stabilization).

Auth and attribution:

- Phase 1a uses the shared `TTS_API_KEY`.
- Dashboard sends `X-Caller: cfo-dashboard-events` on each request.
- Per-adapter keys are Phase 2.
- Browser never receives or stores the key.

Recommended dashboard env knobs after approval:

- `DASHBOARD_TTS_ENABLED=0|1`.
- `DASHBOARD_TTS_URL=http://127.0.0.1:8050`.
- `DASHBOARD_TTS_API_KEY=<shared service key>` (or whatever the launcher already provides).
- `DASHBOARD_TTS_MAX_CHARS=220`.
- `DASHBOARD_TTS_EVENT_ALLOWLIST=sync_all_summary` (single value — loosening requires its own Rule 23 pass).

Failure mode (Rule 23 #9):

- If `/api/events/speak` returns non-200, the browser keeps normal visual behavior (toasts, tile messages).
- Default fallback is text-only toast/feed note. No browser Web Speech API fallback. No fallback to Voicebox in client code (Voicebox is failover at the operator level, not in the dashboard request path).
- Backend logs `[TTS] caller=cfo-dashboard-events event=sync_all_summary status=<int> err=<short>` and returns HTTP 503 to the browser if `voice_tts` is unreachable; browser swallows the 503 silently.
- Spoken text and bearer tokens NEVER appear in any log line.

Cache leverage and wording:

- Sync-all narration is one of exactly two canonical strings. Cache hit on `voice_tts` LRU is nearly guaranteed after warmup.
- The 220-char cap is well above the canonical string lengths; the shared helper still runs every call as the single enforcement point.

Latency decision:

- Mike accepts 500-1200 ms for new short dashboard phrases.
- No streaming in Phase 1a.

### 3.3 Files CFO dashboard would touch

No files are changed in this design pass. If Mike approves implementation after Rule 23 round 3, likely changes:

- `/home/msr8109/projects/cfo_pipeline/dashboard/dashboard.py`
  - Add TTS env config constants near existing env reads.
  - Add backend helper `synthesize_sync_all_summary(ok: int, total: int, failed_items: list[str]) -> bytes | None`. Helper builds the canonical speech string server-side from structured fields, never from caller text.
  - Add proxy/event route `POST /api/events/speak` that accepts ONLY the structured schema in §3.2; rejects any free-form `text` field with HTTP 400.
  - Use shared `TTS_API_KEY` and send `X-Caller: cfo-dashboard-events`.
  - Import shared `trim_for_speech` from the new `voice_text_utils` module (see §4).
  - Add speaker unlock/mute control HTML inside the existing `link-cluster` div near line 3296. No new topbar surface.
  - Hook `setupSyncAll()` at lines 3219-3223 (in the existing `setTimeout` block) to write structured fields to `sessionStorage.cfoLastSyncSummary` immediately before `window.location.reload()`.
  - On page load, read + atomically clear `sessionStorage.cfoLastSyncSummary`, then call `POST /api/events/speak` with the structured fields (NO text) when audio is unlocked.
  - Add inline JS for audio unlock, `localStorage` preference, mute toggle, and playback.
  - Hook ONLY the sync-all final summary. HIGH leak narration is removed from Phase 1a; do NOT add any leak event hook here.
  - Do not hook MED/LOW events, individual Plaid/QBO/QBWC pings, dashboard Search Chat (CFO `/semantic-search` route at line 5736 stays untouched), or unrelated dashboard tiles.
- `/home/msr8109/projects/cfo_pipeline/scripts/daily_sync.py`
  - **No changes in Phase 1a.** The existing Telegram alert path for HIGH/MED leaks stays exactly as-is. Phase 1a does not POST from `daily_sync.py` to `/api/events/speak`. There is no `cfo.leak_events.tts_pending` column, no next-visit queue, no offline narration playback. HIGH leak narration is deferred to Phase 1a.5 (§9.1) and will have its own design pass.
- `/home/msr8109/projects/cfo_pipeline/tests/test_repair_regressions.py` or a new test file
  - Assert TTS API key is never present in rendered HTML.
  - Assert `/api/events/speak` rejects any request body containing a `text`, `narration`, or `message` field with HTTP 400.
  - Assert `/api/events/speak` rejects events outside the `["sync_all_summary"]` allowlist (e.g. `leak_high`, `leak_med`, `plaid_link`) with HTTP 400.
  - Assert `/api/events/speak` rejects malformed `ok`/`total` (negative, non-int, `ok > total`) with HTTP 400.
  - Assert backend constructs the canonical speech string from structured fields and that the canonical string is what is forwarded to `/speak`.
  - Assert backend fails closed (HTTP 503, ERROR log) when `DASHBOARD_TTS_API_KEY` / `TTS_API_KEY` is empty/unset.
  - Assert current Plaid fetch paths remain intact.
  - Assert CFO `/semantic-search` route at line 5736 is untouched by Phase 1a.
  - Assert no `daily_sync.py` code path POSTs to `/api/events/speak` (negative test: HIGH leak path is unchanged).
- CFO dashboard process environment/startup wrapper
  - Add `DASHBOARD_TTS_*` env vars after locating the actual launcher/service definition.

### 3.4 New dependencies

- No required Python dependency: CFO dashboard already has `httpx` and `requests` in `.venv`.
- No browser dependency: use native `Audio`, `Blob`, and `URL.createObjectURL`.
- No ffmpeg needed.

### 3.5 Risks

- Browser autoplay: must require Mike's click to unlock audio; otherwise narration will appear broken.
- Secret leakage: never put TTS key in JS, HTML, query strings, or browser-visible headers.
- Voicebox dual-running: Voicebox stays up during Phase 1a; do not kill it.
- Audio buffering: WAV files can be hundreds of KB; sync-all canonical phrases are short, so this is bounded.
- Reload double-speak: the page must clear `sessionStorage.cfoLastSyncSummary` atomically before issuing the `/api/events/speak` request. Otherwise a second reload mid-fetch could double-narrate. Acceptance test required.
- Dashboard complexity: `dashboard.py` is 5793 lines and inline. Implementation should be minimal and tested, not a frontend rewrite.
- Allowlist drift: `/api/events/speak` must hard-reject any event outside `["sync_all_summary"]`. Loosening (e.g. adding leak_high in Phase 1a.5) requires its own Rule 23 review pass.
- Trust boundary: `/api/events/speak` must NEVER accept caller-supplied speech text. Implementation must include the negative test that rejects `text`/`narration`/`message` fields with HTTP 400 (§3.3).
- Cross-doc risk: hub/app.py event narration is owned by Phase 1b, not Phase 1a. Reviewers must confirm Phase 1a touches only `cfo_pipeline/dashboard/dashboard.py` and not `hub/app.py`.
- Cloudflare: public/browser route behavior is Phase 2, not Phase 1a.

### 3.6 Verification plan

Per-event matrix after implementation approval (sync-all only — HIGH leak narration is out of Phase 1a scope per R2-3):

| Test | Setup | Expected |
|---|---|---|
| `/healthz` baseline | Start CFO dashboard unchanged except adapter env | `GET /healthz` still returns `{"ok": true, "service": "cfo-dashboard"}`. |
| No key in browser | Fetch `/` HTML | HTML contains no TTS token value and no `Authorization: Bearer`. |
| Audio locked | Load CFO dashboard, do not click speaker | Visual `addStatus()` and `tile-sync-msg` still work; no audio request/play attempt. |
| Audio unlock | Click speaker control | Browser marks audio enabled; no console autoplay error. |
| Plaid success narration | Stub or controlled success response | Visual `addStatus(...)` still appears; no Phase 1a audio (Plaid pings stay visual). |
| Sync-all success (ok==total) | Trigger sync-all via `sync-all-btn` with all items succeeding | After reload, one canonical summary phrase `CFO dashboard online. Plaid sync complete.` is spoken. `sessionStorage.cfoLastSyncSummary` is cleared. |
| Sync-all partial (ok<total) | Trigger sync-all with N items failing | After reload, one canonical phrase `Plaid sync failed for N item(s). Review the dashboard.` is spoken. |
| Reject caller text | POST `/api/events/speak` with `{"event":"sync_all_summary","text":"<injected>"}` | HTTP 400. No call to `/speak`. |
| Reject other event types | POST `/api/events/speak` with `{"event":"leak_high","event_id":42}` or any non-allowlisted event | HTTP 400. No call to `/speak`. |
| Reject malformed counts | POST with `ok=-1`, `total=0`, `ok > total`, or non-integer values | HTTP 400. No call to `/speak`. |
| TTS down | Point CFO dashboard TTS URL at unused port in a test process only | Visual UI unaffected; backend logs `[TTS] caller=cfo-dashboard-events ... status=...`; browser sees HTTP 503; no console crash. |
| Adapter fail-closed | Start dashboard with `DASHBOARD_TTS_API_KEY` and shared `TTS_API_KEY` both unset | ERROR log on first attempt: `[TTS] caller=cfo-dashboard-events skipped reason=missing_key`. Endpoint returns HTTP 503. No unauthenticated call to `/speak`. |
| Overlong canonical | Force a canonical template longer than 220 chars in a unit test | Route enforces cap via shared `trim_for_speech`; never sends raw text. |
| Phase 1.5 isolation | CFO `/semantic-search` page renders | Unchanged behavior; no TTS, no RAG hook here. |
| Phase 1b isolation | `hub/app.py` greeting and search chat | Unchanged by Phase 1a deploy (separate file). |
| daily_sync.py unchanged | Run daily_sync HIGH leak path | Telegram alert fires as before; no `/api/events/speak` POST; no `tts_pending` column written. |
| Regression | Existing Plaid Link, account expanders, event pages, sync-all behavior | Existing fetch and render behavior unchanged. |

## 4. Shared speech truncation helper (Rule 23 #5, #11)

A new shared module is required because three call sites need identical truncation behavior:

- JAX `synthesize_voice()` (cap 300).
- CFO dashboard `/api/events/speak` (cap 220).
- Phase 1b `hub/app.py` event narration (cap 220, defined in Phase 1b doc).
- Phase 1.5 `rag_core` voice_text generation (cap 220, defined in Phase 1.5 doc).

Two divergent implementations would drift. Rule 23 cursor §5 mandates one shared spec.

### 4.1 Location

Phase 1a adds a small standalone module:

```text
/home/msr8109/projects/voice_text_utils/__init__.py
/home/msr8109/projects/voice_text_utils/trim.py
/home/msr8109/projects/voice_text_utils/tests/test_trim.py
```

It is intentionally NOT placed inside `voice_tts` because `voice_tts` is the synth service; callers do their own trimming before the call.

It is also intentionally NOT placed inside JAX or dashboard because either of those would make the other adapter import across project boundaries. A small dedicated package is the cleanest seam.

### 4.2 Public API

```python
def trim_for_speech(text: str, cap: int, *, strip_citations: bool = False) -> str:
    """
    Deterministic spoken-text truncation.

    Steps:
      1. If strip_citations: remove inline source IDs matching r"\[[a-z0-9_]+:[a-z0-9_\.-]+\]".
      2. Collapse runs of whitespace to single spaces and strip leading/trailing whitespace.
      3. If len(text) <= cap, return text.
      4. Split into sentences using a punctuation regex matching the end of `.`, `!`, `?`,
         followed by whitespace or end-of-string. Unicode-aware.
      5. Accumulate sentences in order; stop as soon as the accumulated length would exceed cap.
      6. If at least one full sentence fits, return that accumulation.
      7. If no sentence fits (sentence longer than cap, or no terminal punctuation found),
         fall back to word-boundary trim: take the longest prefix ending at a whitespace
         boundary whose length is <= cap. If even the first word exceeds cap, hard-trim at
         exactly cap characters.
    """
```

### 4.3 Determinism guarantees

- Pure function. No I/O. No randomness. No locale-dependent collation.
- Same `(text, cap, strip_citations)` triple always produces the same output.
- No silent over-long output. The return value satisfies `len(result) <= cap`.
- No truncation in the middle of a multi-byte UTF-8 sequence (Python str length is codepoints, not bytes).

### 4.4 Tests

Round 2 finding (codex R2-6): the round-1 test list contained an internally inconsistent case. With `cap=30`, "Sentence one. Sentence two." (each ≈14 chars, total ≈28 with the space) BOTH fit under cap, so a test expecting "only first sentence" contradicts the helper's stated greedy-prefix rule. The list below is the corrected, internally consistent set.

Decision for `cap <= 0`: **return empty string** (do not raise). Rationale: callers wrap `/speak` calls; raising forces every call site to handle an exception for a trivially recoverable case, and `""` triggers the existing "skip TTS on empty text" branch every adapter already has. This is the documented contract.

The shared module ships with `tests/test_trim.py` covering exactly:

1. **Empty string** — `trim_for_speech("", cap=300)` → `""`.
2. **Single short sentence under cap** — `trim_for_speech("Hello world.", cap=300)` → `"Hello world."` (unchanged).
3. **Single long sentence over cap** — `trim_for_speech("This is one very long sentence that has no internal punctuation and exceeds the cap.", cap=30)` → word-boundary truncation at or below cap (no mid-word split, `len(result) <= 30`).
4. **Two sentences both fit** — `trim_for_speech("Sentence one. Sentence two.", cap=200)` → `"Sentence one. Sentence two."` (full string returned because both fit).
5. **Two sentences, first fits, second would exceed cap** — `trim_for_speech("Sentence one. Sentence two is much longer and pushes us past the cap entirely.", cap=20)` → `"Sentence one."` (greedy prefix of full sentences that stays under cap).
6. **No punctuation at all** — `trim_for_speech("alpha beta gamma delta epsilon zeta eta theta iota kappa", cap=25)` → word-boundary truncation, `len(result) <= 25`, no mid-word split.
7. **`cap <= 0`** — `trim_for_speech("anything", cap=0)` → `""`; `trim_for_speech("anything", cap=-5)` → `""`. Documented contract: empty string, not raise.
8. **Whitespace-only input** — `trim_for_speech("   \t  \n  ", cap=300)` → `""` (collapse + strip yields empty).

Additional invariants exercised across the suite (not extra cases — properties asserted within the cases above plus a small property test):

- Output length invariant: `len(result) <= cap` for every output where `cap > 0`.
- Determinism / idempotence: `trim_for_speech(trim_for_speech(x, cap=220), cap=220) == trim_for_speech(x, cap=220)` for the cases above.
- Unicode safety: at least one case uses non-ASCII codepoints (e.g. `"café."`) and asserts the result never splits a codepoint.
- Citation stripping (when `strip_citations=True`, used by Phase 1.5 only): one case with an inline `[ldrive_docs:abc]` token asserts the bracket span is removed before length is measured.

JAX, CFO dashboard, hub/app.py (Phase 1b), and `rag_core` (Phase 1.5) all import `voice_text_utils.trim_for_speech`. No per-caller reimplementation.

### 4.5 Versioning

Phase 1a ships v0.1. Phase 1b reuses v0.1. Phase 1.5 reuses v0.1 with `strip_citations=True`. Any spec change requires a Rule 23 amendment to this section.

## 5. Production hardening (Rule 23 #6)

### 5.1 Enforce TTS_API_KEY at voice_tts startup (Rule 23 round 2 R2-1)

The current behavior in `voice_tts/server.py:111` allows `TTS_API_KEY` to be unset, which bypasses bearer auth on `/voices`, `/speak`, and `/speak/stream`. Round 1 cursor §1 and codex §3 flagged this. Round 2 codex/cursor both flagged the round-1 fix as still unsafe because it relied on `ENVIRONMENT=production` — a misconfigured or unset `ENVIRONMENT` could leave the bypass behavior intact.

**New rule (round 2 — REPLACES the round-1 ENVIRONMENT-based logic; no ENVIRONMENT-based conditional logic anywhere):**

1. `voice_tts` REQUIRES `TTS_API_KEY` to be set by default on every bind interface (loopback `127.0.0.1` / `::1` AND any non-loopback). There is no bind-based carve-out; the only way to run without a key is the explicit dev/test escape in bullet 2.
2. The only way to run `voice_tts` without `TTS_API_KEY` is to set `ALLOW_UNAUTH_TTS=1` explicitly in the service environment. This flag is documented as a **dev/test escape hatch** and is NOT for production use.
3. If `TTS_API_KEY` is unset/empty AND `ALLOW_UNAUTH_TTS != 1`, the service MUST refuse to start. It logs a loud, single-line ERROR — `[TTS-STARTUP] refusing to start: TTS_API_KEY is unset and ALLOW_UNAUTH_TTS!=1` — and exits with non-zero status before the XTTS model is loaded (no GPU allocation).
4. If `ALLOW_UNAUTH_TTS=1` is set explicitly, the service emits a loud WARNING — `[TTS-STARTUP] WARNING: running with unauthenticated /speak; ALLOW_UNAUTH_TTS=1 set explicitly` — and serves anyway. This preserves local-dev ergonomics without ambiguity.
5. There is no `ENVIRONMENT` check, no `production` keyword, no implicit dev/prod inference. The escape hatch is one explicit env var.

Phase 1a operational policy:

- Production `voice_tts` Docker compose file (and any production launcher) MUST set `TTS_API_KEY` from an env file or secret. `ALLOW_UNAUTH_TTS` MUST be unset / `0` in production.
- JAX, CFO dashboard, hub (Phase 1b), and rag_core (Phase 1.5) all read the SAME `TTS_API_KEY` value (shared across Phase 1a/1b).
- Per-adapter keys are Phase 2.
- This rule applies to both `127.0.0.1` and non-loopback bindings; the "non-loopback" trigger above is the production guard, but the dev-only escape hatch is still gated by `ALLOW_UNAUTH_TTS=1` even on loopback to keep the contract uniform.

Production hardening summary:

- Service refuses to start without the key. No silent bypass. No ambiguous environment names.
- Adapters refuse to call without the key (see §2.2 and §3.2 fail-closed preconditions).
- Spoken text and bearer tokens never appear in log output (see §1 service-side `X-Caller` handling).

### 5.2 X-Caller attribution (Rule 23 round 2 R2-5)

Phase 1a header convention for all callers, ALIGNED to the fixed server-side allowlist in §1 (`["jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub"]`):

| Caller | `X-Caller` value |
|---|---|
| JAX Telegram bot | `jax` |
| CFO dashboard event narration | `cfo-dashboard-events` |
| MASTER DASHBOARD event narration (Phase 1b) | `hub` |
| MASTER DASHBOARD Search Chat readback (Phase 1.5) | `dashboard-rag` |
| Smoke tests | `smoke` |

Round-2 change: `voice_tts` NOW reads `X-Caller` server-side (see §1). The value is sanitized against the allowlist above; anything else is normalized to `"unknown"` for logging only. Every `/speak` and `/speak/stream` call writes one structured log line `[TTS] caller=<sanitized> status=<int> latency_ms=<n>`. The spoken text body and the bearer token MUST NEVER appear in log output. Adapter-side logging continues independently for parity.

Note on round-1 names: `dashboard-events` and `hub-events` are RETIRED. Any adapter that emitted those values is required to update to the new names before round 3 review.

## 6. Failure cascade (Rule 23 #9)

Phase 1a follows a strict cascade. Across all adapters:

1. Primary: `voice_tts` on `127.0.0.1:8050`.
2. On `voice_tts` failure: degrade to text-only / visual-only. Log loudly. Do not page Mike.
3. NEVER re-route a failed TTS call to a cloud TTS (Anthropic, OpenAI, etc.).
4. NEVER re-route a sensitive answer to a cloud LLM as a fallback. This rule binds Phase 1.5 / `rag_core`; Phase 1a only generates short canonical/event strings, but the rule is documented here for uniformity.

Implementation guidance:

- Every adapter logs `[TTS] caller=<x-caller> status=<http or err> latency_ms=<n>` (adapter-side). `voice_tts` ALSO logs the same shape server-side (§1, §5.2).
- Every adapter has a kill switch env var (`JAX_TTS_MODE=off`, `DASHBOARD_TTS_ENABLED=0`).
- Every adapter is fail-closed on a missing local TTS key env var (§2.2, §3.2). Never call `/speak` without an `Authorization: Bearer ...` header.
- Voicebox stays as an operator-level failover for Phase 1a. It is NOT in the adapter request path.

## 7. Phase 1.5 isolation

Phase 1a CFO dashboard work is event narration only. It must not modify:

- CFO `/semantic-search` page or backend (line 5736 area).
- MASTER DASHBOARD Search Chat (`hub/app.py` `/api/search-chat`, owned by Phase 1.5).
- Qdrant, Ollama, Anthropic, or any retrieval/synthesis logic.

Allowed in Phase 1a CFO dashboard adapter:

- Sync-all final voice summary only.
- Backend `/api/events/speak` endpoint (structured-fields-only; server-derived canonical speech; see §3).
- Browser audio unlock/mute control for sync-all summary playback only.

Not allowed in Phase 1a CFO dashboard adapter:

- Search Chat changes.
- Semantic search changes.
- RAG pipeline changes.
- Qdrant retrieval changes.
- New chat prompts or answer synthesis.
- Any broad dashboard notification rewrite beyond the event narration allowlist.
- MASTER DASHBOARD (`hub/app.py`) edits of any kind.

## 8. Cross-cutting concerns

Shared cache:

- JAX, CFO dashboard, and (in Phase 1b) hub all hit the same `/speak` service on `127.0.0.1:8050`.
- Common short phrases warm the LRU for all adapters.
- No negative interaction expected as long as adapters send bounded text and use the canonical `voice="mike"`, `language="en"`.
- Unique long JAX replies can churn the shared LRU; JAX defaults to opt-in and max 300 chars.

Observability (Phase 2):

- Phase 2 should add a lightweight `/metrics` endpoint or structured logs in `voice_tts` for request count by caller, status count, synth latency histogram, cache hit/miss count, bytes generated, and active voice/model status. Not required for Phase 1a.

Security:

- Keep `TTS_API_KEY` in backend process env only.
- Never expose to browser JS, Telegram messages, logs, docs, or prompts.
- Do not route `voice_tts` to Windows or public Cloudflare in Phase 1a.

## 9. Phase 2 deferred

This design covers only PGX-local adapters: JAX on PGX and the PGX-hosted CFO dashboard backend proxying to PGX-local `voice_tts`. It does NOT create:

- A Windows-reachable voice surface.
- A Cloudflare route for `voice_tts`.
- A systemd unit for adapter daemons.
- A browser extension or tray app.
- A global desktop voice.

Those require a public/remote-reachable endpoint, additional authentication, browser autoplay UX validation, and Rule 23 triple-review before any code or infrastructure changes land.

Also deferred:

- OGG/Opus native Telegram voice notes.
- Streaming `/speak/stream` adoption.
- Per-adapter `TTS_API_KEY`.
- Voicebox retirement.
- Cache observability and Prometheus metrics.

### 9.1 Phase 1a.5 — HIGH leak narration (deferred from Phase 1a)

HIGH leak narration was scoped in Phase 1a round 1. Rule 23 round 2 (both reviewers, R2-3) flagged that delivering audio for a HIGH leak to an active browser session requires a real-time browser delivery channel (SSE, WebSocket, or short-interval polling) that is not in scope for Phase 1a. Without that channel, the only options are (a) queue narration to play on a later browser visit — which Mike rejected in §13 round-1 decision #2 because it produces stale audio — or (b) silently drop the audio whenever no browser is unlocked, which makes the feature unreliable enough that it should not ship in this phase.

Phase 1a.5 will design HIGH leak narration as its own Rule 23 envelope. Scope outline (to be detailed in `ADAPTER_DESIGN_PHASE1A_5.md` when Mike wants it):

- Choose a live delivery channel: SSE (`/events/stream`), WebSocket, or short-interval polling tuned for one user.
- Decide whether the trigger is browser-pull (page polls a backend "pending narrations" endpoint) or server-push.
- Keep the same server-side speech construction discipline: caller text rejected; server builds speech from the DB-backed `cfo.leak_events` row keyed by `event_id`.
- Define dedupe-by-`event_id` policy so a single HIGH leak does not narrate repeatedly when the browser refocuses or reconnects.
- Reuse the same `trim_for_speech(cap=220)` helper from §4.
- Reuse the same `cfo-dashboard-events` `X-Caller` value and the same fail-closed precondition.

Until Phase 1a.5 ships, `cfo.leak_events` HIGH severity rows are delivered exactly as today: visual surfaces on `/events` + `/events/{event_id}`, plus the existing `scripts/daily_sync.py` Telegram alert path. No `tts_pending` column. No `/api/events/speak` integration for leaks.

## 10. Mike decisions locked for Phase 1a (revised after Rule 23 round 2)

1. JAX trigger policy: opt-in only via `/voice <prompt>` or `voice: <prompt>`. Default JAX behavior is text-only.
2. JAX Telegram format: WAV audio attachment first. OGG/Opus is Phase 2.
3. JAX scope: normal conversational replies only when `/voice` is invoked. If Mike explicitly invokes `/voice` on a search query, JAX may speak the reply; auto-voice on search remains NO.
4. CFO dashboard event scope: **sync-all final summary ONLY**. HIGH leak narration is removed from Phase 1a (deferred to Phase 1a.5; see §9.1). MED/LOW leak events and individual Plaid/QBO/QBWC pings stay visual-only. The CFO allowlist for Phase 1a is exactly `["sync_all_summary"]`; loosening requires its own Rule 23 pass.
5. CFO dashboard audio default: remember enabled state in `localStorage` after first unlock. Visible mute toggle always present, placed inside the existing `link-cluster` div near line 3296.
6. Voicebox migration: keep Voicebox as failover during Phase 1a. Retirement is a separate Rule 23 decision (combined with Phase 1b stabilization).
7. Auth model: shared `TTS_API_KEY` for Phase 1a. Use `X-Caller: jax` and `X-Caller: cfo-dashboard-events` for attribution. Per-adapter keys are Phase 2. Adapters are fail-closed on missing local TTS key (§2.2, §3.2).
8. Length caps: JAX `300`, CFO dashboard `220`. Enforced through the shared `trim_for_speech` helper (§4). No per-caller reimplementation. `cap <= 0` returns empty string.
9. Latency budget: 500-1200 ms is acceptable for new short phrases. No streaming in Phase 1a.
10. Wording: server-derived only for the CFO event path. `/api/events/speak` rejects caller-supplied text (HTTP 400) and constructs the canonical speech string from structured fields per §3.2. JAX speaks the Claude reply text the user already received in chat.
11. Production: `voice_tts` requires `TTS_API_KEY` by default; the only way to run without is explicit `ALLOW_UNAUTH_TTS=1` (dev/test escape hatch). No `ENVIRONMENT`-based logic anywhere (§5.1).
12. `voice_tts` reads and sanitizes `X-Caller` server-side, logging `[TTS] caller=... status=... latency_ms=...`. Spoken text and bearer tokens are NEVER logged (§1, §5.2).
13. Sync-all narration timing: speak after the existing reload completes, using structured fields persisted via `sessionStorage.cfoLastSyncSummary` and cleared atomically on read. No queue, no next-visit replay, no `tts_pending` DB column.
14. MASTER DASHBOARD event narration is out of scope for Phase 1a (moved to Phase 1b).
15. MASTER DASHBOARD Search Chat / RAG is out of scope for Phase 1a (moved to Phase 1.5).

## 11. Recommended implementation order (revised — minimal scope after round 2)

1. **`voice_text_utils` shared trim module + tests.** Ships first because every adapter depends on it. Zero adapter changes here. Covers the 8 test cases in §4.4 plus the invariants. `cap <= 0` returns empty string (documented contract).
2. **`voice_tts` auth + X-Caller guard.** One PR: (a) startup refuses to start without `TTS_API_KEY` unless `ALLOW_UNAUTH_TTS=1`; no `ENVIRONMENT`-based logic (§5.1). (b) `/speak` and `/speak/stream` read `X-Caller`, sanitize against the fixed allowlist, and log `[TTS] caller=... status=... latency_ms=...` without ever logging text or tokens (§1, §5.2).
3. **JAX opt-in WAV adapter.** Smallest user-visible surface; proves end-to-end TTS in Mike's assistant loop. Fail-closed on missing key (§2.2). `X-Caller: jax`.
4. **CFO dashboard sync-all proxy + UI.** `/api/events/speak` with structured-fields schema and HTTP 400 on caller-supplied text. Server-derived canonical strings. `sessionStorage.cfoLastSyncSummary` reload-replay pattern. Speaker/mute toggle inside the existing `link-cluster` div. Fail-closed on missing key (§3.2). `X-Caller: cfo-dashboard-events`. **No HIGH leak code path is added** — that is Phase 1a.5.
5. **Tests.** All §2.6 and §3.6 cases — including the negative tests that prove `/api/events/speak` rejects caller text, rejects non-allowlisted events, rejects malformed counts, fail-closes on missing key, and proves `daily_sync.py` does not call the endpoint.
6. **Rule 23 round 3.** Triple-review the implementation against the round-2 closure table in §14 and the round-3 reviewer checklist in §12. Land only when 2-of-3 reviewers verdict GO.

Out of scope for Phase 1a (do NOT bundle into this order):

- Phase 1a.5 HIGH leak narration (separate design pass, §9.1).
- Phase 1b (hub/app.py) — follows independently after Phase 1a is GO.
- Phase 1.5 (`rag_core` + hub Search Chat) — follows after Phase 1a + 1b stable.

## 12. Rule 23 review checklist — round 3 (post-round-2 closure)

This checklist is what round-3 reviewers must verify after the round-2 must-fix items are closed in this doc (round 2 source: `docs/rule23/round2/SUMMARY.md`).

Core round-2 closure (all reviewers must confirm):

1. **All 7 round-2 must-fix items are closed in this doc.** Cross-check against the §14 closure table.
2. **HIGH leak narration is fully removed from Phase 1a scope.** No HIGH leak reference remains in §3 trigger policy, §3.3 files-touched, §3.6 verification, or §10 decisions. HIGH leak appears only in §9.1 (Phase 1a.5 deferred) and §14 (closure log).
3. **Adapters fail-closed on missing TTS key.** §2.2 (JAX) and §3.2 (CFO) both contain the hard precondition: log ERROR, return HTTP 503 to caller (CFO) or skip (JAX), and NEVER call `/speak` without `Authorization: Bearer ...`.
4. **`voice_tts` reads and sanitizes `X-Caller` server-side.** §1 and §5.2 describe one structured log line per request with shape `[TTS] caller=<sanitized> status=<int> latency_ms=<n>`. Allowlist is `["jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub"]`; unknown values normalize to `"unknown"`. Spoken text and bearer tokens never appear in any log line.
5. **`/api/events/speak` rejects caller-supplied speech text.** Server-derived canonical strings only. Request bodies containing `text`, `narration`, or `message` fields return HTTP 400 (§3.2 schema + §3.3 test list).
6. **`voice_text_utils` test list is internally consistent.** §4.4 lists exactly the 8 required cases. The two-sentence test no longer contradicts the greedy-prefix rule. `cap <= 0` returns empty string (documented).
7. **CFO allowlist is `["sync_all_summary"]` only.** Verify `/api/events/speak` rejects any other event value with HTTP 400. Verify `daily_sync.py` does not POST to `/api/events/speak` (negative test required, §3.3).

Cursor-specific round-3 checks:

- Phase 1a touches only `jax-bot/jax_bot.py`, `cfo_pipeline/dashboard/dashboard.py`, the new `voice_text_utils/` package, and `voice_tts/server.py` (startup guard + `X-Caller` handling).
- Phase 1a does NOT touch `hub/app.py`, `qdrant_master_search/master_search_api.py`, `scripts/daily_sync.py`, or any Phase 1.5 file.
- No new `window.alert()`; no leaked secrets; no test fixture key.
- Auth rule uses `ALLOW_UNAUTH_TTS=1` only; no `ENVIRONMENT` checks anywhere in the diff.
- Shared trim module covered by tests including word-boundary fallback and `cap <= 0` (§4.4).
- `TTS_API_KEY` startup guard verified by a failing-test (service refuses to start; no GPU allocation).

Codex-specific round-3 checks:

- Phase 1a / Phase 1a.5 / Phase 1b / Phase 1.5 scope separation is preserved in the diff.
- `voice_tts` API usage matches the live contract in §1.
- Failure cascade is loud in logs but non-disruptive (§6).
- Adapter fail-closed behavior verified end-to-end (no `/speak` call when key is empty).
- `X-Caller` sanitization is a hard allowlist match, not a substring match.
- No Cloudflare, systemd, APEX/CFO production restart, or Telegram token handling change beyond approved scope.

Perplexity Sonar round-3 checks (when reviewer is usable; round 2 banner-failed):

- Telegram Bot API audio vs voice-note constraints and file size limits.
- Chrome autoplay policy and unlock pattern.
- Security posture for same-origin backend proxy hiding TTS bearer token.
- Whether `sessionStorage.cfoLastSyncSummary` atomic clear is sufficient against fast double-reload race in current Chrome.

## 13. Codex-delegated decisions (round 1) — status after round 2

Mike delegated five decisions to Codex on 2026-05-18 for the revised round-1 Rule 23 pass. Round 2 reshaped scope; status of each is now:

1. **`TTS_API_KEY` startup guard ships inside Phase 1a (not as a separate pre-Phase-1a patch).** Status: **still binding**, with the round-2 revision in §5.1 — guard now uses `ALLOW_UNAUTH_TTS=1` instead of `ENVIRONMENT`. Still the first code change and first verification checkpoint in §11.
2. **HIGH leak playback semantics: live browser session only, no next-visit queue.** Status: **OBSOLETE.** HIGH leak narration is removed from Phase 1a entirely (round 2 R2-3). Phase 1a.5 will redesign HIGH leak delivery; until then there is no Phase 1a HIGH leak playback at all, live or queued. See §9.1.
3. **Shared helper location at `/home/msr8109/projects/voice_text_utils/` as a sibling package.** Status: **still binding.** §4.1 is unchanged.
4. **CFO mute/speaker toggle inside the existing `link-cluster` div near line 3296; no new topbar surface.** Status: **still binding.** §3.2 / §10 #5.
5. **Sync-all summary timing: speak after reload via `sessionStorage.cfoLastSyncSummary`; not before `window.location.reload()`.** Status: **still binding and is now the sole CFO narration path.** Round-2 contract in §3.2 keeps the same reload-replay pattern, with the added discipline that the browser writes structured fields (counts, optional failed-item slugs), the backend reads + atomically clears that key on next load and constructs the canonical speech string server-side. No caller text. If audio is not unlocked after reload, show the visual summary and leave voice silent.

## 14. Rule 23 closure round 2

This section maps the round-2 must-fix items from `docs/rule23/round2/SUMMARY.md` to the specific revisions in this doc that close them. Source verdicts: codex `NEEDS-CHANGES` (7 numbered items) and cursor `CONDITIONAL` (3 must-fix items). Both reviewers agreed on the 3 convergent must-fix items. Perplexity Sonar banner-failed (not usable, same as round 1).

| Round-2 item (SUMMARY.md §) | Revision(s) in this doc that close it |
|---|---|
| **R2-1** TTS_API_KEY auth bypass risk (convergent; SUMMARY § "Convergent #1") | §0 round-2 list (R2-1 entry); §1 auth contract note removing `ENVIRONMENT` ambiguity; §5.1 rewritten: require key by default, only `ALLOW_UNAUTH_TTS=1` dev/test escape hatch, no `ENVIRONMENT`-based logic anywhere. |
| **R2-2** `/api/events/speak` accepts raw caller text (convergent; SUMMARY § "Convergent #2") | §3 header note; §3.2 trigger policy and data flow rewritten to server-derived canonical strings from structured fields; §3.2 request schema with HTTP 400 for any `text`/`narration`/`message`; §3.3 negative tests; §3.6 verification matrix rows "Reject caller text" and "Reject other event types". |
| **R2-3** HIGH leak playback architecturally incomplete (convergent; SUMMARY § "Convergent #3") | §3 header note removing HIGH leak from Phase 1a; §3.2 trigger policy lists only sync-all; §3.3 explicitly NOT changing `daily_sync.py`, removing `tts_pending` and next-visit queue logic; §9.1 new Phase 1a.5 deferred subsection with rationale (needs SSE/polling, separate design); §10 #4 narrowed; §13 #2 marked OBSOLETE. |
| **R2-4** Adapter fail-closed on missing TTS key (codex-only; SUMMARY § "Codex #4") | §2.2 JAX hard precondition block; §3.2 CFO hard precondition block; §6 implementation guidance bullet; §12 round-3 checklist item 3; §3.6 "Adapter fail-closed" verification row. |
| **R2-5** Service-side X-Caller handling in `voice_tts` (codex-only; SUMMARY § "Codex #5") | §1 new X-Caller handling block (sanitization allowlist, single structured log line shape, no text/token in logs); §5.2 retitled and aligned to the new server-side allowlist (`jax`, `cfo-dashboard-events`, `dashboard-rag`, `smoke`, `hub`); §6 logging bullet now says adapter AND service log; §12 round-3 checklist item 4. |
| **R2-6** `voice_text_utils` test inconsistency (codex-only; SUMMARY § "Codex #6") | §4.4 rewritten with the 8 required test cases plus invariants; the previously contradictory `cap=30 / "Sentence one. Sentence two."` example replaced by case #5 ("Sentence two is much longer ..."); explicit `cap <= 0` decision documented (returns empty string, does not raise). |
| **R2-7** Keep CFO allowlist tight (codex-only; SUMMARY § "Codex #7") | §3 header lock-in (`["sync_all_summary"]` only); §3.2 trigger policy ends after sync-all; §3.2 env-knob `DASHBOARD_TTS_EVENT_ALLOWLIST=sync_all_summary` (single value); §3.5 risk "Allowlist drift" bullet; §10 #4 narrowed; §12 round-3 checklist item 7; "loosening requires its own Rule 23 pass" stated in §3 header, §3.2, §10 #4, and §3.5. |

Round-2 cursor-only nice-to-haves (not blocking; tracked for future):

- Optional dedupe of HIGH leak narration by `event_id` — folded into Phase 1a.5 design checklist in §9.1.
- "Safe default" for `voice_tts` on PGX — addressed by §5.1's new rule: key required by default whenever the listener binds non-loopback; no implicit `production` inference.

Health snapshot from round 2 (per `docs/rule23/round2/SUMMARY.md`):

- voice_tts: healthy, voices=["mike"].
- voicebox: healthy (legacy failover).
- jax-bot (screen `jax` per CLAUDE.md Rule 2): active.
- No source code modified during round 2 review.
- No Docker / Cloudflare / systemd changes during round 2.

This revision pass continues that posture: design-only changes to this file. Round 3 (post-revision triple-review) is the next gate before any implementation may land.
