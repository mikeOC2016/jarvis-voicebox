# Voice TTS Adapter Design ? Phase 1 (PGX-Local)

Date: 2026-05-18
Author: Codex
Gate status: DESIGN ONLY, pre-Rule-23 review. No implementation approved. No code, Docker, systemd, Cloudflare, JAX, or dashboard changes are included in this document.

## 1. Service contract recap

The `voice_tts` service is running on PGX only at `127.0.0.1:8050`, backed by XTTS-v2 and a single canonical reference voice named `mike`.

Current health evidence from `GET /health`:

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

Endpoints from `~/projects/voice_tts/server.py`:

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| `GET` | `/health` | none | none | JSON liveness/model state. Returns `status: "ok"` after model load; otherwise `"loading"`. |
| `GET` | `/voices` | bearer | none | `{"voices": ["mike"]}`. |
| `POST` | `/speak` | bearer | JSON `{"text": str, "voice": str = "mike", "language": str = "en"}`; `text` min 1 max 2000 chars | `audio/wav` body, 24 kHz mono PCM WAV. Headers: `X-Synthesis-Ms`, `X-Voice`, `X-Language`, `X-Sample-Rate`, `X-Audio-Bytes`, `Cache-Control: no-store`. |
| `POST` | `/speak/stream` | bearer | same schema as `/speak` | `audio/wav` `StreamingResponse` with WAV header and PCM chunks. Headers: `X-Voice`, `X-Language`, `X-Sample-Rate`, `Cache-Control: no-store`, `X-Accel-Buffering: no`, `CF-Cache-Status: BYPASS`. |

Auth contract:

- All endpoints except `/health` use `Authorization: Bearer <TTS_API_KEY>`.
- If `TTS_API_KEY` is not set in the service process, auth is bypassed by current code. Production/adapters should not rely on bypass.
- Missing bearer token returns `401` with detail `missing bearer token`.
- Wrong bearer token returns `403` with detail `invalid token`.

Request validation and failure modes:

- `/speak` and `/speak/stream` use Pydantic `SpeakRequest` with `text` length `1..2000`, default `voice="mike"`, default `language="en"`.
- Unknown voice raises `HTTPException(404, "voice not found: <voice>")`.
- Model missing/not loaded raises `503 "model not loaded"`.
- Service startup catches fatal model-load exceptions, leaves `_ready = False`, and Docker healthcheck should mark it unhealthy.

Cache semantics:

- `server.py` normalizes text first with `normalize_tts_text(text)`.
- Cache key is `xxh3_64(normalized_text + NUL + voice + NUL + language)`.
- Two requests cache-hit only when normalized text, voice, and language match exactly.
- `/speak` uses the LRU audio cache. `/speak/stream` does not currently populate or read that audio cache.
- LRU capacity is `LRU_CAPACITY` env var, default `1000`. Eviction is oldest-entry first via `OrderedDict.popitem(last=False)`.
- `audio_cache_size` in `/health` is the number of cached complete clips currently held in memory, not disk.

Latency profile from recent verification:

- Warm cache smoke test after restart: all three smoke phrases returned `HTTP 200`; repeated/cached responses reported `X-Synthesis-Ms: 0` with single-digit total elapsed milliseconds in prior checks.
- Cold/new text after text normalization: operational phrase around `1.5s`, numbers-heavy phrase around `4.0s`, because normalized numbers expand the spoken text substantially.
- Earlier cold smoke numbers were around `489ms` to `1187ms` depending on phrase length.
- `/speak/stream` is the preferred path when first-byte latency matters, but Phase 1 adapters should start with `/speak` unless Mike chooses otherwise.

Service `.env` contract from `.env.example` and `server.py`:

| Env var | Meaning | Current design use |
|---|---|---|
| `TTS_API_KEY` | Bearer token expected by `/voices`, `/speak`, `/speak/stream` | Adapters should read their own copy from env, or dashboard should proxy so browser never sees it. |
| `DEFAULT_VOICE` | Default voice if request omits `voice` | Should remain `mike`. |
| `DEFAULT_LANG` | Default language if request omits `language` | Should remain `en`. |
| `LRU_CAPACITY` | Max complete clips in in-memory LRU | Shared across JAX/dashboard; default 1000. |
| `LOG_LEVEL` | Python logging level | Keep `INFO` for adapter rollout. |
| `WARMUP_TEXT` | Startup warmup phrase | Current warmup creates at least one cache entry. |
| `VOICES_DIR` | Reference voice mount path, default `/voices` | No adapter impact. |
| `CACHE_DIR` | Model cache mount path, default `/cache` | No adapter impact. |
| `XTTS_MODEL` | Model name, default XTTS-v2 | No adapter impact. |

## 2. Adapter A - JAX - Telegram voice notes

### 2.1 Current state

JAX root: `/home/msr8109/jax-bot`.

Entry point: `/home/msr8109/jax-bot/jax_bot.py`, launched as `/usr/bin/python3 /home/msr8109/jax-bot/jax_bot.py`.

Framework:

- `python-telegram-bot` 22.7.
- Uses `telegram.ext.Application`, `CommandHandler`, `MessageHandler`, `filters`, and `ContextTypes`.
- Polling mode: `app.run_polling(drop_pending_updates=True, allowed_updates=["message"])`.
- Existing HTTP client dependency: `httpx` 0.28.1 is already imported and installed through `python-telegram-bot`/system user packages.

Token isolation:

- Current code contains `BOT_TOKEN` with token prefix `8798832090` in `jax_bot.py`.
- This design does not propose changing, moving, logging, rotating, or otherwise touching that token.
- Any implementation must preserve Rule 27 token isolation and should not print the full token in logs or reviews.

Config pattern:

- No `requirements.txt`, `pyproject.toml`, `.env`, or external config file was found in `/home/msr8109/jax-bot`.
- JAX currently uses module-level constants in `jax_bot.py`: `BOT_TOKEN`, `ALLOWED_USER_ID`, `CLAUDE_BIN`, `WORKDIR`, `CLAUDE_TIMEOUT`, `TG_LIMIT`, search endpoints, and L-drive paths.
- For TTS, the least disruptive pattern is to add env-var reads with safe defaults, not introduce a full config system in Phase 1.

Current response pipeline:

1. `handle_message(update, context)` receives text messages from the allowed Telegram user.
2. It logs the incoming text to `/home/msr8109/jax-bot/jax.log`.
3. It sends `ChatAction.TYPING`.
4. It routes:
   - `ping` -> direct `msg.reply_text("pong")`.
   - `read/open/summarize/send N` follow-ups -> `handle_search_followup()`.
   - file-search intents -> `handle_file_search()`.
   - all other text -> `run_claude(text)` using `claude -p`, then reply chunks.
5. Long text is split by `chunk_text()` to stay under Telegram text limits.
6. Replies are sent with `msg.reply_text(...)`; files use `context.bot.send_document(...)`.

Telegram send call sites in active `jax_bot.py`:

- `handle_file_search`: line 319 `await msg.reply_text(piece)`.
- `_send_file`: lines 342, 348, 351, 356, 362, 385, 387 `reply_text`; lines 368-376 `context.bot.send_document(...)`.
- `handle_search_followup`: lines 397, 402, 411, 422, 431, 458, 467, 471 `reply_text`.
- `keep_typing`: line 480 `bot.send_chat_action(...)`.
- `start_cmd`: line 519 `update.message.reply_text(...)`.
- `handle_message`: line 542 `send_chat_action`, line 545 ping `reply_text`, lines 578-579 Claude reply chunks `reply_text`.
- `ignore_unauth` only logs.

The best adapter seam is not every call site. It is a small helper called only after final user-visible text is produced. Start with the Claude reply path at `handle_message` lines 567-580 and optionally direct `ping` later. File-search lists and document summaries are noisy and should be opt-in.

### 2.2 Proposed integration

Mike decision for Phase 1:

- Opt-in only.
- Triggers: `/voice <prompt>` command or `voice: <prompt>` prefix.
- Default JAX behavior remains text-only.
- WAV audio attachment first; OGG/Opus native Telegram voice notes are deferred to Phase 2.
- Scope: normal conversational replies only when `/voice` or `voice:` is invoked. No automatic speaking of file-search results or document summaries.
- Clarification: opt-in `/voice` means the user explicitly asked for voice. If Mike invokes `/voice` on a search query, JAX may speak whatever reply that path returns. Auto-voice on search/document paths remains NO.

Data flow:

```text
Telegram message
  -> JAX detects /voice or voice: prefix
  -> JAX strips trigger and routes remaining prompt through the existing pipeline
  -> JAX sends the full text reply normally
  -> JAX chooses TTS text:
       if reply <= 300 chars: speak full reply
       if reply > 300 chars: speak first 2 sentences only
  -> POST http://127.0.0.1:8050/speak
       Authorization: Bearer <shared TTS_API_KEY>
       X-Caller: jax
       JSON {"text": tts_text, "voice": "mike", "language": "en"}
  -> receive WAV bytes
  -> send WAV audio attachment to Telegram
```

Telegram message format:

- Phase 1 uses WAV audio attachment (`send_audio` preferred if Telegram accepts the file object cleanly; `send_document` fallback if needed).
- OGG/Opus native voice-note bubbles are Phase 2 because they require an ffmpeg/Opus conversion path and temp-file cleanup.
- Do not block Phase 1 on voice-note UX; first prove TTS value in the JAX loop.

Auth and attribution:

- JAX uses the shared `TTS_API_KEY` for Phase 1.
- JAX sends `X-Caller: jax` on every TTS request.
- Per-adapter keys are Phase 2 if needed.
- The Telegram bot token with prefix `8798832090` is not moved, changed, printed, rotated, or involved in the TTS auth path.

Recommended JAX env knobs after approval:

- `JAX_TTS_MODE=off|command`, default `command` after rollout and `off` for emergency kill switch.
- `JAX_TTS_URL=http://127.0.0.1:8050`.
- `JAX_TTS_API_KEY=<shared service key>`.
- `JAX_TTS_VOICE=mike`.
- `JAX_TTS_LANG=en`.
- `JAX_TTS_MAX_CHARS=300`.

Failure mode:

- Text-first always.
- If `/speak` returns non-200, times out, or raises, JAX logs `[TTS] failed caller=jax status=... err=...` and continues text-only.
- TTS failure must never suppress or delay the normal text reply beyond the explicit voice attachment attempt.
- User-visible TTS failure messages should stay off by default unless Mike asks for them.

Length cap:

- `voice_tts` accepts up to 2000 chars, but Phase 1 JAX cap is 300 chars for spoken text.
- If the final JAX text reply is over 300 chars, JAX sends the full text in Telegram as usual and speaks only the first two sentences.
- If the first two sentences are still over 300 chars, trim at a sentence or word boundary and log `[TTS] truncated caller=jax chars_in=N chars_spoken=M`.

Opt-out / kill switch:

- `JAX_TTS_MODE=off` disables all voice without changing code.
- With `JAX_TTS_MODE=command`, only `/voice` and `voice:` trigger TTS.
- No persistent per-chat state in Phase 1.

Latency decision:

- Mike accepts 500-1200 ms for new short JAX phrases.
- No streaming in Phase 1.
- Streaming is Phase 2 only if later UX requires faster first byte.

### 2.3 Files JAX would touch

No files are changed in this design pass. If Mike approves implementation after Rule 23, likely JAX changes are:

- `/home/msr8109/jax-bot/jax_bot.py`
  - Add env-var config constants for TTS.
  - Add trigger parsing for `/voice <prompt>` and `voice: <prompt>`.
  - Add `async synthesize_voice(text: str) -> bytes | None` using existing `httpx.AsyncClient`.
  - Add a delivery helper `send_tts_audio(update, context, text)`.
  - Use shared `TTS_API_KEY` and send `X-Caller: jax`.
  - Call the helper only after the normal text reply on explicitly voiced paths.
  - Enforce `JAX_TTS_MAX_CHARS=300` with first-two-sentences spoken for longer replies.
  - Add loud log lines for skipped/truncated/failed/succeeded TTS calls.
- `/home/msr8109/jax-bot/test_file_search.py` or a new test file in the same directory
  - Existing tests are minimal; add pure tests for trigger parsing, search-query opt-in behavior, and length-cap behavior without hitting Telegram or TTS.
- JAX process environment/startup wrapper, if one exists outside this directory
  - Add `JAX_TTS_*` env vars. Investigation did not find a local `.env`; implementation must locate the actual launcher before editing it.

### 2.4 New dependencies

For WAV audio attachment:

- No new Python dependency. JAX already has `httpx` and `python-telegram-bot`.
- Uses standard library `io`/`tempfile` if needed.

For native Telegram voice notes:

- Requires system `ffmpeg` with Opus support, or a Python audio encoding package.
- Recommended not to add this in the first implementation; verify WAV path first.

### 2.5 Risks

- Token isolation: JAX token prefix `8798832090` must not be changed, printed, rotated, or moved in this adapter pass.
- Telegram file limits: Bot API file upload limit is commonly 50 MB for bot uploads; current short WAVs are hundreds of KB, but long replies could grow fast.
- Telegram UX: WAV audio attachment is less polished than voice note; OGG/Opus voice note requires extra encoding machinery.
- Rate limits: Telegram send calls plus TTS generation can add latency; text-first avoids perceived JAX failure.
- Claude output length: Long answers should not be auto-voiced.
- Audio cache memory: Repeated common phrases cache well; unique long JAX replies will churn the shared LRU.
- Temp files: If Telegram API helpers require file-like objects, implementation must clean temp files even on exceptions.
- TTS service outage: Must degrade to text-only, loudly logged.

### 2.6 Verification plan

Read-only design; no verification executed for an adapter yet. Proposed implementation verification:

1. Unit test trigger parsing:
   - `voice hello` -> voice requested, prompt `hello`.
   - `/voice hello` -> voice requested, prompt `hello`.
   - normal text with `JAX_TTS_MODE=command` -> no voice.
   - over `JAX_TTS_MAX_CHARS` -> skip with reason `too_long`.
2. Dry-run TTS helper test with a fake HTTP client:
   - 200 with WAV bytes -> returns bytes and headers.
   - 401/403/404/503 -> returns `None`, logs error.
   - timeout -> returns `None`, logs error.
3. Manual Telegram test in a controlled JAX session:
   - Send `/voice JAX bot ready. Standing by.`
   - Expect normal text reply first.
   - Expect WAV audio attachment second.
4. Failure test without service restart:
   - Temporarily point `JAX_TTS_URL` at an unused local port in a test process only.
   - Send `/voice test`.
   - Expect text reply; expect no crash; expect `[TTS] failed` in JAX log.
5. Regression test:
   - Existing `ping`, file search, `read N`, `open N`, `summarize N`, and `send N` still behave text/file-only unless explicitly voiced.

## 3. Adapter B - MASTER DASHBOARD event narration

### 3.1 Current state

Dashboard root: `/home/msr8109/repos/cfo_pipeline`, symlinked from `~/projects/cfo_pipeline`.

Entry point and serving:

- Active process: `/home/msr8109/projects/cfo_pipeline/.venv/bin/python dashboard/dashboard.py`.
- Framework: FastAPI 0.136.0 with Uvicorn 0.44.0.
- App object: `app = FastAPI()` in `dashboard/dashboard.py`.
- Direct run: `uvicorn.run(app, host="0.0.0.0", port=8098)`.
- Current listener: `0.0.0.0:8098`.
- Browser-facing public route is `cfo.luxqozp.com`, but Phase 1 design is PGX-local only and does not alter Cloudflare.

Existing dashboard event/notification surfaces:

- Plaid Link status messages use inline JS `addStatus(msg, type)` and an auto-removing DOM node near the Link Accounts button.
  - Examples: `Re-link complete for <institution>. Accounts and balances refreshed.`, `Exchange failed...`, `openPlaidLink error...`.
- Institution tiles use `.tile-sync-msg` for sync progress and result messages.
  - Examples: `Syncing 1/N...`, `Synced.`, `Sync failed for N item(s).`, `No active item to sync.`.
- Leak events are persisted in `cfo.leak_events` and displayed under `/events` and `/events/{event_id}`.
  - Fields surfaced: `event_type`, `severity`, `merchant_norm`, `annualized_impact`, `message`, `acknowledged`.
  - Event types observed in code: `DUPLICATE`, `PRICE_HIKE`, `ZOMBIE`, `ENTITY_MISALLOC`, `WEBHOOK`.
- `scripts/daily_sync.py` sends Telegram text alerts for high/medium severity leak events through `send_telegram()`, and marks `alerted=TRUE`.
- There is no active SSE, WebSocket, or continuous browser polling in `dashboard/dashboard.py`.
- Browser interactions are fetch-on-click: Plaid link/exchange, account transaction expansion, item sync, sync-all.

Current Voicebox/TTS integration:

- Active CFO dashboard search excluding `.venv` and backups found no `voicebox`, `VOICEBOX`, `17493`, `voice_tts`, `127.0.0.1:8050`, `/speak`, `send_voice`, or `send_audio` call sites.
- Voicebox container is running separately at `127.0.0.1:17493`, but no active dashboard call site was found to migrate in this pass.
- Therefore the Phase 1 dashboard adapter is a new event narration surface, not a direct replacement of active dashboard Voicebox code.

Existing dependencies/config:

- `dashboard/dashboard.py` already imports `requests` and `httpx as _httpx_pi`.
- `.venv` has `httpx`, `requests`, FastAPI, Uvicorn, psycopg2, and Plaid installed.
- Config pattern is direct `os.environ[...]` / `os.environ.get(...)` at module import time. `.env` exists but was not printed because it can contain secrets.

### 3.2 Proposed integration

Mike decision for Phase 1:

- Dashboard speaks HIGH leak events and sync-all final summary only.
- MED/LOW leak events remain visual only.
- Individual Plaid, QBO, and QBWC pings remain visual only.
- Allowlist can expand later after the initial adapter is stable.
- Dashboard remembers audio-enabled state in `localStorage` after first unlock.
- A visible mute/unmute toggle is always present.
- Voicebox stays running as failover during Phase 1. Retirement is a separate Rule 23 decision after two weeks of stable adapter usage.

Phase 1 trigger policy:

1. Sync-all final summary:
   - Speak one final summary only after the existing sync-all loop completes.
   - Do not speak per-item progress.
   - Candidate canonical strings:
     - `CFO dashboard online. Plaid sync complete.`
     - `Plaid sync failed for <N> item(s). Review the dashboard.`
2. HIGH leak event narration:
   - Speak only HIGH severity leak events.
   - Dynamic content is allowed because signal matters more than cache hit rate for content events.
   - Candidate format: `High CFO leak alert. <event message>`.

Data flow:

```text
Dashboard event occurs
  -> dashboard backend validates event type against allowlist
  -> dashboard backend builds short narration string (<= 220 chars)
  -> POST http://127.0.0.1:8050/speak
       Authorization: Bearer <shared TTS_API_KEY>
       X-Caller: dashboard-events
       JSON {"text": narration, "voice": "mike", "language": "en"}
  -> dashboard backend returns/proxies WAV to browser as same-origin audio
  -> browser plays only if audio is unlocked and not muted
```

Browser autoplay constraint:

- Chrome blocks autoplaying audio until the page has a user gesture.
- Dashboard uses an unlock pattern:
  - Always show a visible speaker/mute toggle in the topbar.
  - First click unlocks audio and stores enabled state in `localStorage`.
  - After first unlock, the dashboard may remember enabled state across sessions, but the visible mute toggle remains available.
  - Until unlocked/enabled, dashboard keeps visual toasts only and does not treat blocked audio as an error.
- No attempt should be made to bypass browser autoplay policy.

Audio delivery design:

- Browser must not call `voice_tts` directly because `voice_tts` is bound to `127.0.0.1` on PGX and the browser cannot reach it from Mike's Windows machine through `cfo.luxqozp.com` without a proxy.
- Browser must never see `TTS_API_KEY`.
- Preferred design: dashboard backend proxy/event endpoint:
  - Backend validates event type and text length.
  - Backend calls `http://127.0.0.1:8050/speak` with `Authorization: Bearer <shared TTS_API_KEY>` and `X-Caller: dashboard-events`.
  - Backend returns `audio/wav` to browser with `Cache-Control: no-store`.
- For sync-all, prefer canonical fixed phrases for cache hits.
- For HIGH leak events, include dynamic merchant/event names when they carry useful signal.

Voicebox migration:

- Active dashboard call sites to migrate: none found.
- Keep Voicebox container running as failover during Phase 1.
- Do not retire or stop Voicebox in this phase.
- Voicebox retirement requires separate Rule 23 review after two weeks of stable adapter usage.

Auth:

- Phase 1 uses the shared `TTS_API_KEY`.
- Dashboard sends `X-Caller: dashboard-events` on each request.
- Per-adapter keys are deferred to Phase 2 if needed.
- Browser never receives or stores the key.

Recommended dashboard env knobs after approval:

- `DASHBOARD_TTS_ENABLED=0|1`.
- `DASHBOARD_TTS_URL=http://127.0.0.1:8050`.
- `DASHBOARD_TTS_API_KEY=<shared service key>` or shared `TTS_API_KEY` if the launcher already carries it.
- `DASHBOARD_TTS_MAX_CHARS=220`.
- `DASHBOARD_TTS_EVENT_ALLOWLIST=sync_all_summary,leak_high`.

Failure mode:

- If TTS is down or rejects a request, dashboard keeps normal visual behavior.
- Default fallback is text-only toast/feed note, not browser Web Speech API.
- Browser Web Speech API uses a different voice and is not part of Phase 1.
- Backend logs `[TTS] caller=dashboard-events event=... status=... err=...`.

Cache leverage and wording:

- Hybrid wording policy:
  - Use canonical fixed strings for high-frequency events such as sync complete and future kill-switch style events.
  - Use dynamic names/details for content events where the details matter, such as a HIGH leak event.
- Do not sacrifice important event signal purely for cache hit rate.
- Keep spoken event strings short and under the 220-char cap.

Latency decision:

- Mike accepts 500-1200 ms for new short dashboard phrases.
- No streaming in Phase 1.
- Streaming is Phase 2 only if dashboard real-time feel requires it.

### 3.3 Files dashboard would touch

No files are changed in this design pass. If Mike approves implementation after Rule 23, likely dashboard changes are:

- `/home/msr8109/repos/cfo_pipeline/dashboard/dashboard.py`
  - Add TTS env config constants near existing env reads.
  - Add backend helper `synthesize_dashboard_tts(text, event_type)` using existing `httpx` or `requests`.
  - Add proxy/event route for dashboard narration audio.
  - Use shared `TTS_API_KEY` and send `X-Caller: dashboard-events`.
  - Add topbar speaker unlock/mute control HTML in `render()`.
  - Add inline JS for audio unlock, `localStorage` preference persistence, mute toggle, and playback of backend-proxied WAV.
  - Hook only selected event paths: sync-all final summary and HIGH leak event narration.
  - Do not hook MED/LOW events, individual Plaid/QBO/QBWC pings, dashboard Search Chat, or unrelated dashboard tiles.
- `/home/msr8109/repos/cfo_pipeline/tests/test_repair_regressions.py` or a new test file
  - Assert API key is not present in rendered HTML.
  - Assert proxy/event route enforces text length and allowlist.
  - Assert current Plaid fetch paths remain intact.
  - Assert dashboard Search Chat/RAG code paths are untouched in Phase 1.
- Dashboard process environment/startup wrapper
  - Add `DASHBOARD_TTS_*` env vars after locating the actual launcher/service definition.

### 3.4 New dependencies

- No required Python dependency: dashboard already has `httpx` and `requests` in `.venv`.
- No browser dependency: use native `Audio`, `Blob`, and `URL.createObjectURL`.
- No ffmpeg needed for dashboard Phase 1 because browser can play WAV from the backend proxy.

### 3.5 Risks

- Browser autoplay: must require Mike's click to unlock audio; otherwise narration will appear broken.
- Secret leakage: never put TTS key in JS, HTML, query strings, or browser-visible headers.
- Voicebox dual-running: Voicebox can stay up during migration; do not kill it in Phase 1. Later retirement requires separate gate.
- Audio buffering: WAV files can be hundreds of KB; browser should play only short event phrases and avoid long narration.
- Event spam: sync loops can produce repeated messages; frontend should debounce/queue and avoid overlapping audio.
- Dashboard complexity: `dashboard.py` is large and inline; implementation should be minimal and tested, not a frontend rewrite.
- Cloudflare: public/browser route behavior needs Rule 23 and is Phase 2, not this PGX-local design.

### 3.6 Verification plan

Per-event matrix after implementation approval:

| Test | Setup | Expected |
|---|---|---|
| `/healthz` baseline | Start dashboard unchanged except adapter env | `GET /healthz` still returns `{"ok": true, "service": "cfo-dashboard"}`. |
| No key in browser | Fetch `/` HTML | HTML contains no TTS token value and no `Authorization: Bearer`. |
| Audio locked | Load dashboard, do not click speaker | Visual toasts still work; no audio request/play attempt. |
| Audio unlock | Click speaker control | Browser marks audio enabled; no console autoplay error. |
| Plaid success narration | Stub or controlled success response | Visual `addStatus(...)` still appears; browser fetches proxied WAV; audio plays. |
| Sync-all success | Trigger sync-all in test/stub path | One summary phrase, not one phrase per item. |
| TTS down | Point dashboard TTS URL at unused port in a test process only | Visual UI unaffected; backend logs `[TTS] failed`; browser sees no crash. |
| Overlong event | Send text over cap to proxy route | Route rejects or skips gracefully; no call to `voice_tts`. |
| Leak high/med event | Use existing `/events` data or test fixture | Narrates one short summary only when enabled. |
| Regression | Existing Plaid Link, account expanders, event pages | Existing fetch and render behavior unchanged. |

## 4. Phase 1.5 isolation - Dashboard Search Chat / RAG boundary

Phase 1 dashboard work is event narration only. It must not modify dashboard Search Chat, semantic search, RAG, Qdrant query behavior, Search Chat UI, or any chat answer-generation path.

Phase 1.5 is a separate future design lane for upgrading Search Chat to RAG. That work will need its own read-only investigation, design doc, Mike GO, and Rule 23 review. Keeping this boundary explicit prevents the event narration adapter from becoming a dashboard AI/search rewrite.

Allowed in Phase 1 dashboard adapter:

- Sync-all final voice summary.
- HIGH leak event narration.
- Backend TTS proxy/event endpoint needed only for those event narration paths.
- Browser audio unlock/mute control needed only for event narration playback.

Not allowed in Phase 1 dashboard adapter:

- Search Chat changes.
- Semantic search changes.
- RAG pipeline changes.
- Qdrant retrieval changes.
- New chat prompts or answer synthesis.
- Any broad dashboard notification rewrite beyond the event narration allowlist.

## 5. Cross-cutting concerns

Shared cache:

- JAX and dashboard should both hit the same `/speak` service on `127.0.0.1:8050`.
- This is positive: common phrases warm the LRU for all adapters.
- No negative interaction expected as long as adapters send bounded text and use the canonical `voice="mike"`, `language="en"`.
- Unique long JAX responses can churn the shared LRU, so JAX should default to opt-in and max 300 chars.

Caller attribution:

- Proposed request header convention for all adapters:
  - `X-Caller: jax`
  - `X-Caller: dashboard`
  - `X-Caller: smoke`
- Current `voice_tts` does not read or log this header. Phase 2 service observability can add it.
- Until then, adapters should log their own outbound TTS calls with caller/event/status/latency.

Observability:

- Phase 2 should consider a lightweight `/metrics` endpoint or structured logs for:
  - request count by caller
  - status count
  - synth latency histogram
  - cache hit/miss count
  - bytes generated
  - active voice and model status
- Prometheus is useful later, but not required for Phase 1 adapters.

Disk/memory growth:

- `voice_tts` audio cache is in-memory LRU with default capacity 1000 clips, not disk.
- Model cache under `cache/` is persistent and expected.
- Generated test output and quality-test WAVs are disk artifacts; define a cleanup policy before long-term use.

Debug artifact cleanup:

- Current canonical voice surface is `voices/mike.wav` plus `voices/raw/` source material.
- `voices/raw/mike_raw.wav` is untouchable source.
- `voices/raw/_prep/` or prep/debug artifacts, if present later, should be retained only when needed for audit; generated comparison WAVs under `quality_tests/` and `test_outputs/` can be pruned by explicit gated cleanup.

Security:

- Keep TTS API keys in backend process env only.
- Never expose keys to browser JS, Telegram messages, logs, docs, or prompts.
- Do not route `voice_tts` to Windows or public Cloudflare in Phase 1.

## 6. Phase 2 deferred (Cloudflare, Windows tray, Chrome extension)

This design intentionally covers only PGX-local adapters: JAX on PGX and the PGX-hosted CFO dashboard backend proxying to PGX-local `voice_tts`. It does not create a Windows-reachable voice surface, Cloudflare route, systemd unit, browser extension, tray app, or global desktop voice. Those require a public/remote-reachable endpoint, additional authentication decisions, browser autoplay UX validation, and Rule 23 triple-review before any code or infrastructure changes land.

## 7. Mike decisions locked for Phase 1

1. JAX trigger policy: opt-in only through `/voice <prompt>` or `voice: <prompt>`. Default JAX behavior is text-only.
2. JAX Telegram format: WAV audio attachment first. OGG/Opus native Telegram voice notes are deferred to Phase 2.
3. JAX scope: normal conversational replies only when `/voice` is invoked. No automatic speaking of file-search results or document summaries. Clarification: if Mike explicitly invokes `/voice` on a search query, JAX may speak whatever reply the search path returns; auto-voice on search remains NO.
4. Dashboard event scope: HIGH leak events and sync-all final summary only. MED/LOW leak events and individual Plaid/QBO/QBWC pings stay visual-only. Allowlist can expand later.
5. Dashboard audio default: remember enabled state in `localStorage` after first unlock. Visible mute toggle always present.
6. Voicebox migration: keep Voicebox as failover during Phase 1. Retirement is a separate Rule 23 decision after two weeks of stable adapter usage.
7. Auth model: shared `TTS_API_KEY` for Phase 1. Use `X-Caller: jax` and `X-Caller: dashboard-events` for attribution. Per-adapter keys are Phase 2 if needed.
8. Length caps: JAX `300`, dashboard `220`. If a JAX reply is over 300 chars, speak the first two sentences and send the full text in Telegram as usual.
9. Latency budget: 500-1200 ms is acceptable for new short phrases. No streaming in Phase 1. Streaming is Phase 2 if dashboard real-time feel requires it.
10. Wording: hybrid. Use canonical fixed strings for high-frequency events such as sync complete and kill-switch style events. Use dynamic names for content events where signal matters. Do not sacrifice useful signal solely for cache hit rate.

## 8. Recommended implementation order

1. JAX opt-in WAV adapter first. It is the smallest surface: one process, existing `httpx`, existing Telegram file-send capability, and text-first fallback. It proves TTS quality in Mike's actual assistant loop without touching Cloudflare or the dashboard.
2. Add JAX OGG/Opus voice-note support later only if Mike rejects WAV audio attachments. This isolates the ffmpeg/Opus decision after the core TTS call path works.
3. Dashboard backend proxy/event endpoint third. This solves the browser-to-PGX-local boundary correctly by keeping `TTS_API_KEY` server-side and gives the browser a same-origin audio URL.
4. Dashboard audio unlock and visible mute toggle fourth. Browser autoplay is the riskiest UX piece; build it after the backend proxy can return WAV reliably.
5. Dashboard event allowlist fifth. Start with only sync-all final summary and HIGH leak events, per Mike's decision.
6. Observability and cleanup sixth. Add `X-Caller` logging/metrics and a cleanup policy after both adapters generate real usage.
7. Only after the above is stable, run Rule 23 for Phase 2 surfaces: Cloudflare, Windows tray/global voice, Chrome extension, service units, streaming, OGG/Opus voice notes, per-adapter keys, Search Chat/RAG, or retiring Voicebox.

## 9. Rule 23 review checklist (for later)

Cursor review should verdict on:

- Exact JAX diff: trigger parsing, text-first fallback, no token movement, no full token logging.
- Exact dashboard diff: no browser-exposed secret, existing Plaid/dashboard behavior preserved, no `window.alert()` added.
- Tests for trigger parsing, length cap, TTS failure fallback, and dashboard proxy auth/allowlist.

Perplexity Sonar review should verdict on:

- Telegram Bot API audio vs voice-note constraints and file size limits.
- Chrome autoplay policy and unlock pattern.
- Security posture for same-origin backend proxy hiding TTS bearer token.
- Operational risk of keeping Voicebox running as fallback vs retiring it later.

Codex review should verdict on:

- Whether adapter scope is still PGX-local and within Mike's gate.
- Whether `voice_tts` API usage matches the actual service contract.
- Whether cache semantics and length caps are sane for XTTS-v2.
- Whether failure modes are loud in logs but non-disruptive to JAX/dashboard users.
- Whether any change touches Cloudflare, systemd, APEX/CFO production restart, or Telegram token handling beyond the approved scope.
