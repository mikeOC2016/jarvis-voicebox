# Voice TTS Adapter Design — Phase 1b (MASTER DASHBOARD event narration)

Date: 2026-05-18
Author: Claude (new doc following Path 2 split after Rule 23 NEEDS-CHANGES)
Companion docs: `ADAPTER_DESIGN_PHASE1A.md` (JAX + CFO), `ADAPTER_DESIGN_PHASE1_5.md` (MASTER DASHBOARD Search Chat / RAG)
Gate status: DESIGN ONLY. No implementation approved. No code, Docker, systemd, Cloudflare, JAX, dashboard, or `voice_tts` changes are part of this document.

## 0. Why this doc exists

Rule 23 cursor + codex flagged that the original Phase 1 design mixed two different dashboards. CFO Pipeline `dashboard.py` and MASTER DASHBOARD `hub/app.py` are separate FastAPI applications with different surfaces, different event models, and different existing voice integrations.

Path 2 (Mike's decision) splits the dashboard adapter into:

- **Phase 1a** — JAX adapter + CFO event narration in `cfo_pipeline/dashboard/dashboard.py` (HIGH leak events, sync-all summary).
- **Phase 1b** — MASTER DASHBOARD event narration in `hub/app.py` (greeting on dashboard load, optional service-card status transitions). Migrate the existing legacy Jarvis Voicebox call path (`127.0.0.1:8110`) to `voice_tts` (`127.0.0.1:8050`) with Voicebox kept as operator-level failover.
- **Phase 1.5** — MASTER DASHBOARD Search Chat / RAG in `hub/app.py` using a new shared `rag_core` library (Option C).

This doc covers Phase 1b only.

Phase 1b reuses Phase 1a's shared building blocks:

- `voice_text_utils.trim_for_speech()` (Phase 1a §4) for deterministic 220-char speech text.
- `voice_tts` `TTS_API_KEY` production-mode startup guard (Phase 1a §5.1).
- `X-Caller` attribution convention (Phase 1a §5.2), specifically `X-Caller: hub-events` for Phase 1b.
- Failure cascade rules (Phase 1a §6).

This doc does NOT touch `hub/app.py`'s Search Chat code path. That entire lane is owned by `ADAPTER_DESIGN_PHASE1_5.md`.

## 1. Current state — verified by reading hub/app.py end-to-end

`hub/app.py` is 401 lines. Read 1-402 in this design pass.

Service unit:

```ini
/home/msr8109/.config/systemd/user/hub.service
WorkingDirectory=/home/msr8109/projects/hub
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8101
```

App constants (`hub/app.py:17-24`):

```python
APP_NAME = "MASTER DASHBOARD"
STYLE_MARKER = "ORIGIN_STYLE_MASTER_V1"
TIMEOUT = 0.45
LOCAL_TZ = ZoneInfo("America/New_York")
JARVIS_VOICE_URL = "http://127.0.0.1:8110"
QDRANT_SEARCH_URL = "http://127.0.0.1:8100/search"
DQ_SALES_URL = "http://127.0.0.1:8090/api/dq/sales"
DEFAULT_AUTO_SEARCH_QUERY = "SALES TODAY FOR DQS"
```

Service catalog (`hub/app.py:40-58`) — 17 read-only `ServiceCard` rows across five groups: Finance, Operations, Trading, AI Search, Infrastructure.

### 1.1 Events emitted by hub/app.py today

Hub is a read-only, fetch-on-click dashboard. There are no Plaid, QBWC, or leak events. The events that exist are:

1. **Page load greeting** — `build_greeting()` at `app.py:120-142` produces a single sentence describing service health, e.g. `Good evening, Mike. MASTER DASHBOARD is online. 14 of 17 surfaces are online, with 3 needing attention. Jarvis Voice is online.` Triggered automatically by JS at `app.py:322` via `window.setTimeout(playGreeting, 450)`.
2. **Service health probe roll-up** — `collect_status()` at `app.py:81-93` probes every `ServiceCard.probe_url` with a 450 ms timeout per probe (`TIMEOUT = 0.45` at `app.py:19`) and returns a summary `{total, up, down, link}`. Browser auto-refreshes via `setInterval(refreshStatus, 30000)` at `app.py:322`. Today this is silent — the dashboard updates pill text and metric counters in DOM but does not speak status transitions.
3. **`jarvis_text` status indicator** — `build_greeting()` at `app.py:136-140` sets `jarvis_text = "Jarvis Voice is online"` or `"Jarvis Voice is checking in"` based on the live probe of the `jarvis-voice` service card. This is part of the greeting text, not a separate spoken event.

No SSE, no WebSocket, no push notification. All events are pull-based and browser-orchestrated.

### 1.2 Existing `/api/speak` integration — every call site

`synthesize_jarvis_audio(text)` is defined at `app.py:145-158`:

```python
def synthesize_jarvis_audio(text: str) -> tuple[bytes, dict[str, str]]:
    payload = json.dumps({"text": text, "voice_id": "jarvis"}).encode("utf-8")
    req = urllib.request.Request(
        f"{JARVIS_VOICE_URL}/api/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=420) as response:
        data = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    if not data.startswith(b"RIFF"):
        raise RuntimeError("Jarvis Voice did not return WAV audio")
    return data, headers
```

This is the ONE TTS helper in `hub/app.py`. Two server-side call sites call it:

- **A. Greeting playback** — `/api/jarvis/greeting/speak` at `app.py:346-359`:

  ```python
  @app.post("/api/jarvis/greeting/speak")
  def api_jarvis_greeting_speak() -> Response:
      greeting = build_greeting(collect_status())
      try:
          audio, headers = synthesize_jarvis_audio(greeting["text"])
      except Exception as exc:
          raise HTTPException(status_code=502, detail="Jarvis Voice unavailable") from exc
      return Response(audio, media_type="audio/wav", headers={
          "Cache-Control": "no-store",
          "Content-Disposition": 'inline; filename="jarvis_dashboard_greeting.wav"',
          "X-Jarvis-TTS-Provider": headers.get("x-jarvis-tts-provider", "voicebox"),
          "X-Jarvis-TTS-Model": headers.get("x-jarvis-tts-model", "luxtts"),
          "X-Jarvis-TTS-Voice": headers.get("x-jarvis-tts-voice", "JARVIS"),
      })
  ```

  Caller: browser JS `playGreeting()` at `app.py:316`, fired on page load via `window.setTimeout(playGreeting, 450)` at `app.py:322` and on Start/Replay button clicks at `app.py:317-318`.

- **B. Search Chat readback** — `/api/jarvis/speak` at `app.py:362-380`:

  ```python
  @app.post("/api/jarvis/speak")
  async def api_jarvis_speak(request: Request) -> Response:
      payload = await request.json()
      text = str(payload.get("text") or "").strip()
      if not text:
          raise HTTPException(status_code=400, detail="text is required")
      if len(text) > 4096:
          raise HTTPException(status_code=400, detail="text exceeds 4096 characters")
      try:
          audio, headers = synthesize_jarvis_audio(text)
      except Exception as exc:
          raise HTTPException(status_code=502, detail="Jarvis Voice unavailable") from exc
      return Response(audio, media_type="audio/wav", headers={...})
  ```

  Caller: browser JS `runSearch(q)` at `app.py:320` posts `{text: p.answer}` to `/api/jarvis/speak`. This is Search Chat readback, which is Phase 1.5's lane.

Phase 1b's scope is narrower than the helper — only the **greeting** path (call site A above) is a Phase 1b event. **The Search Chat readback (call site B) is owned by Phase 1.5.**

However, both call sites share `synthesize_jarvis_audio()` (the underlying helper at lines 145-158). Phase 1b migrates that shared helper to `voice_tts` and adds Voicebox failover behavior. As a side effect, `/api/jarvis/speak` (call site B) also benefits from the new plumbing. Phase 1.5 retains semantic ownership of `/api/jarvis/speak` and may evolve the request/response contract there independently.

### 1.3 Browser audio-unlock pattern in current index.html

The hub already has an unlock pattern; this matters because Phase 1b does NOT need to invent a new one.

Inline JS (`app.py:303-322`):

```js
const jarvisBootAudio=document.getElementById('jarvisBootAudio');
const jarvisSearchAudio=document.getElementById('jarvisSearchAudio');
const jarvisEnableBtn=document.getElementById('jarvisEnableBtn');
const jarvisReplayBtn=document.getElementById('jarvisReplayBtn');
const jarvisVoiceStatus=document.getElementById('jarvisVoiceStatus');
const jarvisGreetingText=document.getElementById('jarvisGreetingText');
const jarvisVoiceLabel=document.getElementById('jarvisVoiceLabel');
let pendingBootUrl=null;
...
async function playGreeting(){ ... pendingBootUrl=await fetchVoice('/api/jarvis/greeting/speak'); try{await playUrl(jarvisBootAudio,pendingBootUrl); ...}catch(e){jarvisVoiceStatus.textContent='Voice is ready. Click Start Jarvis to play.';} ... }
jarvisEnableBtn.addEventListener('click',async()=>{if(pendingBootUrl){try{await playUrl(jarvisBootAudio,pendingBootUrl); ... return;}catch(e){...}}await playGreeting();});
jarvisReplayBtn.addEventListener('click',playGreeting);
document.addEventListener('click',async()=>{if(pendingBootUrl&&jarvisBootAudio.paused){try{await playUrl(jarvisBootAudio,pendingBootUrl); ... pendingBootUrl=null;}catch(e){}}},{once:true});
...
refreshStatus();window.setInterval(refreshStatus,30000);window.setTimeout(playGreeting,450);window.setTimeout(()=>runSearch(DEFAULT_AUTO_SEARCH_QUERY),1300);
```

Pattern summary:

- Two `<audio>` elements live in the rendered HTML: `jarvisBootAudio` (boot greeting, `app.py:302`) and `jarvisSearchAudio` (search chat, also at `app.py:302`). The `<audio>` elements are `playsinline` and use `controls` for search chat / `preload="auto"` for boot.
- Two visible buttons control the greeting: `jarvisEnableBtn` (id `jarvisEnableBtn`, label `Start Jarvis`) and `jarvisReplayBtn` (id `jarvisReplayBtn`, label `Replay greeting`).
- On page load, `playGreeting()` is invoked 450 ms after load. It fetches the WAV, tries to play it, and on Chrome autoplay-block it saves the blob URL in `pendingBootUrl` and prompts the user via `Start Jarvis`.
- Any subsequent user click anywhere on the page (the global `document.addEventListener('click', ..., {once: true})` handler at `app.py:319`) consumes `pendingBootUrl` and plays the greeting.
- There is no `localStorage` persistence today — the unlock state is per-page-load only.

Phase 1b's job is NOT to rewrite this pattern. Phase 1b reuses it, adds `localStorage` persistence to remember Mike's enable preference (matching Phase 1a's CFO dashboard convention), and migrates the underlying TTS provider.

### 1.4 Why hub/app.py differs from CFO dashboard.py

| Axis | `cfo_pipeline/dashboard/dashboard.py` (Phase 1a) | `hub/app.py` (Phase 1b) |
|---|---|---|
| File size | 5793 lines (heavy, inline) | 401 lines (small, inline) |
| Port | 8098 | 8101 |
| Event model | Plaid, QBO, QBWC, leak events, Postgres-driven | Service probe roll-up only, in-memory |
| Existing TTS | none | `JARVIS_VOICE_URL → /api/speak` (legacy port 8110, audio playback already wired) |
| Event triggers | server-side (DB inserts), client-side (sync-all click) | client-side only (page load + 30s probe loop) |
| Sensitivity surface | financial data | service status only |
| Browser unlock | none today (Phase 1a adds) | partially present (Phase 1b extends) |

The implications:

- Phase 1b does NOT need a new `/api/events/speak` style endpoint. It already has `/api/jarvis/greeting/speak` and `/api/jarvis/speak`. Phase 1b reuses those route paths and changes only the TTS provider inside `synthesize_jarvis_audio()`.
- Phase 1b's only added event is "speak the greeting on dashboard load", which already works today. The change is the TTS backend (voice_tts) and the unlock persistence (localStorage). No new event types added.
- Optionally, Phase 1b adds a service-card status-transition speaker (e.g., speak when a card flips from `up` to `down`). This is deferred to §3.4 as a Mike decision because it adds noise.

## 2. Proposed integration

### 2.1 Scope locked

Phase 1b will:

1. Migrate `synthesize_jarvis_audio()` at `hub/app.py:145-158` from `JARVIS_VOICE_URL` (port 8110) → `voice_tts` (`127.0.0.1:8050`) `POST /speak`.
2. Add `voice_tts` bearer auth (`TTS_API_KEY`) and `X-Caller: hub-events` to outbound requests.
3. Apply the shared `voice_text_utils.trim_for_speech(text, cap=220)` (Phase 1a §4) to every text routed to `voice_tts` from hub events.
4. Keep `JARVIS_VOICE_URL` (port 8110) as a per-request failover ONLY for the `/api/jarvis/greeting/speak` event path. Failover is opt-in via env, default ON for the Phase 1b stabilization window.
5. Add `localStorage.hubGreetingAudioEnabled = "0" | "1"` persistence so Mike's enable preference survives reload.
6. Leave Search Chat readback (`/api/jarvis/speak`) untouched at the route level. The underlying `synthesize_jarvis_audio()` migration benefits it as a side effect, but no Phase 1b code touches Search Chat semantics. Phase 1.5 owns Search Chat.

Phase 1b will NOT:

- Add a service-card status-transition speaker (deferred until Mike confirms — see §3.4).
- Add SSE, WebSockets, or push.
- Change the rendered HTML beyond the additional `localStorage` JS plus optional mute button label.
- Touch `/api/search-chat`, `build_search_chat_answer`, `build_dq_sales_answer`, `search_qdrant_master`, or `fetch_dq_sales`.
- Touch any non-`hub` codebase.

### 2.2 Migrated helper — design only

Proposed shape of the migrated helper inside `hub/app.py`:

```python
HUB_TTS_URL = "http://127.0.0.1:8050"
HUB_TTS_API_KEY = os.environ.get("TTS_API_KEY", "")
HUB_TTS_VOICE = os.environ.get("HUB_TTS_VOICE", "mike")
HUB_TTS_LANG = "en"
HUB_TTS_TIMEOUT_S = float(os.environ.get("HUB_TTS_TIMEOUT_S", "30"))
HUB_TTS_FAILOVER_ENABLED = os.environ.get("HUB_TTS_FAILOVER_ENABLED", "1") == "1"
HUB_LEGACY_VOICE_URL = "http://127.0.0.1:8110"  # Voicebox failover; existing constant relocated

def _try_voice_tts(text: str) -> tuple[bytes, dict[str, str]]:
    """Call canonical voice_tts /speak. Raises on failure for failover decision."""
    speech_text = trim_for_speech(text, cap=220)  # voice_text_utils
    if not speech_text:
        raise RuntimeError("hub: empty speech text after trim")
    payload = json.dumps({"text": speech_text, "voice": HUB_TTS_VOICE, "language": HUB_TTS_LANG}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Caller": "hub-events",
    }
    if HUB_TTS_API_KEY:
        headers["Authorization"] = f"Bearer {HUB_TTS_API_KEY}"
    req = urllib.request.Request(f"{HUB_TTS_URL}/speak", data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=HUB_TTS_TIMEOUT_S) as response:
        data = response.read()
        out_headers = {key.lower(): value for key, value in response.headers.items()}
    if not data.startswith(b"RIFF"):
        raise RuntimeError("hub: voice_tts did not return WAV audio")
    return data, out_headers

def _try_legacy_voicebox(text: str) -> tuple[bytes, dict[str, str]]:
    """Existing port-8110 path, used only as failover."""
    payload = json.dumps({"text": text, "voice_id": "jarvis"}).encode("utf-8")
    req = urllib.request.Request(
        f"{HUB_LEGACY_VOICE_URL}/api/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=420) as response:
        data = response.read()
        out_headers = {key.lower(): value for key, value in response.headers.items()}
    if not data.startswith(b"RIFF"):
        raise RuntimeError("hub: legacy voicebox did not return WAV audio")
    return data, out_headers

def synthesize_jarvis_audio(text: str) -> tuple[bytes, dict[str, str]]:
    """Canonical voice_tts first; legacy Voicebox failover if enabled."""
    try:
        return _try_voice_tts(text)
    except Exception as exc:
        log.warning(f"[TTS] caller=hub-events voice_tts failed err={exc!r}")
        if HUB_TTS_FAILOVER_ENABLED:
            try:
                return _try_legacy_voicebox(text)
            except Exception as exc2:
                log.warning(f"[TTS] caller=hub-events legacy voicebox failed err={exc2!r}")
                raise
        raise
```

Key invariants:

- `trim_for_speech(text, cap=220)` is the only producer of speech text for `/speak`. The function lives in `voice_text_utils` (shared module from Phase 1a §4).
- The legacy voicebox path is intentionally NOT subject to the new cap, because Voicebox is failover and Phase 1b does not change its request shape.
- `X-Caller: hub-events` is sent on every primary call to `voice_tts`.
- `TTS_API_KEY` is honored when set; absence triggers Phase 1a §5.1 production-mode startup-guard behavior in `voice_tts` itself. If `voice_tts` rejects with 401, the failover path runs.

### 2.3 Route handlers — minimal changes

`/api/jarvis/greeting/speak` at `app.py:346-359`:

- No signature change.
- Body change: call the new `synthesize_jarvis_audio()` helper above (no caller-side modification needed because the function name stays the same).
- Response headers stay the same. The `X-Jarvis-TTS-Provider` header is now populated from the primary `voice_tts` response when primary wins, and from Voicebox response when failover wins. Both backends already expose compatible headers in observed traffic; Phase 1b sets a new `X-Hub-TTS-Path: voice_tts|voicebox` header so the frontend can render a small status badge.

`/api/jarvis/speak` at `app.py:362-380`:

- No signature change.
- No body change beyond the underlying helper migration. Search Chat readback (Phase 1.5 lane) keeps its existing route contract.
- Phase 1.5 may later replace this handler entirely; Phase 1b does NOT delete it.

### 2.4 Browser unlock persistence

Phase 1b adds `localStorage`-backed unlock state, mirroring Phase 1a's CFO dashboard convention:

```js
const HUB_AUDIO_KEY = "hubGreetingAudioEnabled";

function isHubAudioEnabled() {
  try { return localStorage.getItem(HUB_AUDIO_KEY) === "1"; }
  catch { return false; }
}

function setHubAudioEnabled(on) {
  try { localStorage.setItem(HUB_AUDIO_KEY, on ? "1" : "0"); }
  catch {}
}
```

Behavior:

- On page load, if `isHubAudioEnabled()` is true, `playGreeting()` runs and attempts autoplay. If the browser still blocks (rare after first-click unlock), `pendingBootUrl` is set and the existing global click handler at `app.py:319` consumes it. If autoplay succeeds, no user action needed.
- On page load, if `isHubAudioEnabled()` is false, `playGreeting()` still fetches the WAV (so it is ready), but skips the auto-play attempt and updates the status text to `Click Start Jarvis to enable voice greetings.` First click on `jarvisEnableBtn` sets the flag to `1` and plays.
- The mute toggle is the new `jarvisEnableBtn`/`jarvisReplayBtn` pair already in the markup. A subtle relabel may be appropriate: Start Jarvis → Enable Jarvis when state is off, Mute Jarvis when state is on. Phase 1b proposal: leave button labels alone, add a third small toggle next to them labeled `Mute` that flips the localStorage flag.

### 2.5 Data flow

```text
Browser loads / -> playGreeting() at app.py:316
  -> if isHubAudioEnabled(): try autoplay; else save pendingBootUrl
  -> POST /api/jarvis/greeting/speak (hub backend)

hub backend
  -> build_greeting(collect_status()) -> short greeting text
  -> trim_for_speech(text, cap=220) -> safe spoken text
  -> _try_voice_tts(safe_text):
       POST http://127.0.0.1:8050/speak
       Authorization: Bearer <TTS_API_KEY>
       X-Caller: hub-events
       JSON {"text": safe_text, "voice": "mike", "language": "en"}
       Timeout: 30s (HUB_TTS_TIMEOUT_S)
  -> on success: return audio/wav with X-Hub-TTS-Path: voice_tts
  -> on failure: if HUB_TTS_FAILOVER_ENABLED:
       POST http://127.0.0.1:8110/api/speak (Voicebox legacy)
       JSON {"text": text, "voice_id": "jarvis"}
     return audio/wav with X-Hub-TTS-Path: voicebox
  -> on double failure: return 502 to browser

Browser
  <- audio/wav -> play via existing jarvisBootAudio element
  <- on 502 -> existing error UX ("Voice check failed") + status text "Voice unavailable; visual mode."
```

## 3. Files hub would touch (with line numbers)

No files are changed in this design pass. If Mike approves implementation after Rule 23, the following exact regions of `/home/msr8109/projects/hub/app.py` would be touched.

### 3.1 Lines added near top-of-file imports

`app.py:1-15` (existing imports). Add:

- `import os`
- `import logging` (use `log = logging.getLogger("hub")`)
- `from voice_text_utils import trim_for_speech`  (shared module from Phase 1a §4)

If the new `voice_text_utils` package is not yet importable in the hub's Python environment, the implementation may temporarily vendor a copy of `trim.py` into `hub/voice_text_utils.py`. Either way, function semantics match Phase 1a §4 verbatim.

### 3.2 Lines added/changed for new constants

`app.py:17-24` (existing constants). Add the new TTS constants from §2.2:

- `HUB_TTS_URL`
- `HUB_TTS_API_KEY`
- `HUB_TTS_VOICE`
- `HUB_TTS_LANG`
- `HUB_TTS_TIMEOUT_S`
- `HUB_TTS_FAILOVER_ENABLED`
- `HUB_LEGACY_VOICE_URL` (renamed/relocated from `JARVIS_VOICE_URL`; the old name is removed)

The old `JARVIS_VOICE_URL` at `app.py:21` is removed. Any external code that imported it (Phase 1b verified there is none; this is a fresh search beyond the file) keeps working because there are no external importers — `hub/app.py` is run as `python -m uvicorn app:app`, not imported as a library.

### 3.3 Lines replaced: synthesize_jarvis_audio

`app.py:145-158`:

Replaced with the new `_try_voice_tts`, `_try_legacy_voicebox`, and `synthesize_jarvis_audio` implementations from §2.2. Net code added: ~50 lines. Net code removed: 14 lines (the existing helper body).

### 3.4 Lines optionally extended: route handlers

`app.py:346-359` (`api_jarvis_greeting_speak`):

- Add `X-Hub-TTS-Path` to the response `headers` dict at line 353-358 area.
- Optionally pass `caller_event="greeting"` to a logging line for `[TTS] caller=hub-events event=greeting status=...`.

`app.py:362-380` (`api_jarvis_speak`):

- No semantic change.
- Add `X-Hub-TTS-Path` response header for consistency.
- Logging line `[TTS] caller=hub-events event=speak-passthrough status=...` (Phase 1.5 may overwrite this).

### 3.5 Lines extended: browser JS

`app.py:303-323` (inline JS in `render_dashboard`).

Add:

- `HUB_AUDIO_KEY`, `isHubAudioEnabled`, `setHubAudioEnabled` helpers from §2.4 between the constants block (`app.py:303-311`) and `refreshStatus` (`app.py:312`).
- Modify `playGreeting()` at `app.py:316` so the autoplay attempt is gated on `isHubAudioEnabled()`.
- Modify `jarvisEnableBtn` click handler at `app.py:317` so it calls `setHubAudioEnabled(true)` after successful first play.
- Optionally add a `Mute` toggle that flips the flag; if added, the HTML for it lives next to `jarvisEnableBtn` in `app.py:302`.

`window.setTimeout(playGreeting, 450)` at `app.py:322` stays.

`runSearch(...)` at `app.py:320` is untouched (Phase 1.5 owns it).

### 3.6 Lines untouched

The following sections of `hub/app.py` are explicitly NOT touched by Phase 1b:

- `app.py:27-58` — `ServiceCard` dataclass and `SERVICE_CATALOG` (read-only).
- `app.py:60-93` — `app = FastAPI(...)`, `probe_url`, `collect_status` (Phase 1b consumes their output but does not modify them).
- `app.py:96-117` — `_tone`, `_initials`, `_part_of_day` helpers.
- `app.py:120-142` — `build_greeting` (consumed by Phase 1b but unchanged).
- `app.py:161-279` — Search Chat helpers (`search_qdrant_master`, `fetch_dq_sales`, `_compact_snippet`, `_money`, `_pct`, `_sales_timestamp`, `_is_dq_sales_today_query`, `build_dq_sales_answer`, `build_search_chat_answer`). **Phase 1.5 owns these.**
- `app.py:282-300` — `render_dashboard()` HTML scaffold for service cards (Phase 1b touches only the JS block).
- `app.py:326-344` — `/`, `/healthz`, `/api/status`, `/api/jarvis/greeting` route handlers.
- `app.py:383-401` — `/api/search-chat`, `/favicon.ico`. **Phase 1.5 owns `/api/search-chat`.**

### 3.7 Service env

`/home/msr8109/.config/systemd/user/hub.service` — Mike-approved env var additions in the `[Service]` block, applied via `systemctl --user daemon-reload && systemctl --user restart hub.service` after Mike approves Phase 1b implementation:

```ini
Environment="TTS_API_KEY=<shared key>"
Environment="HUB_TTS_VOICE=mike"
Environment="HUB_TTS_TIMEOUT_S=30"
Environment="HUB_TTS_FAILOVER_ENABLED=1"
```

No Docker change. No Cloudflare change. No code outside `hub/app.py` and the shared `voice_text_utils` package.

### 3.8 Tests

`/home/msr8109/projects/hub/tests/test_hub_tts.py` (new file):

- `test_synthesize_jarvis_audio_calls_voice_tts_first` — mocks both backends; asserts primary path.
- `test_synthesize_jarvis_audio_falls_back_to_voicebox` — primary raises; asserts failover.
- `test_synthesize_jarvis_audio_double_failure_raises` — both raise; assert exception.
- `test_synthesize_jarvis_audio_passes_x_caller` — assert outbound headers include `X-Caller: hub-events`.
- `test_synthesize_jarvis_audio_passes_bearer_when_key_set` — env-set test.
- `test_synthesize_jarvis_audio_no_bearer_when_key_unset_dev_mode` — env-unset test (development; production guard is in `voice_tts` itself per Phase 1a §5.1).
- `test_greeting_speech_text_under_220` — calls `trim_for_speech` is applied; result length ≤ 220.
- `test_search_chat_path_untouched_at_route_level` — POST `/api/search-chat` with a fixture query still returns the same shape as today (Phase 1.5 will update this test).
- `test_browser_html_has_no_tts_api_key` — fetch `/`, assert response body has no `TTS_API_KEY`-shaped string.
- `test_x_hub_tts_path_header_set_on_success` — assert response header present.
- `test_legacy_voicebox_failover_disabled_by_env` — set `HUB_TTS_FAILOVER_ENABLED=0`, primary raises, assert HTTP 502.

## 4. Rule 23 must-fix items resolved by Phase 1b

| Item | Phase 1b resolution |
|---|---|
| #1 keep CFO event narration isolated | Out of scope here; Phase 1a §3 owns CFO. Phase 1b confirms no Phase 1a file is touched. |
| #5 deterministic truncation helper | Reused from Phase 1a §4 (`voice_text_utils.trim_for_speech`). Phase 1b is a CALLER, not a re-implementation. |
| #6 enforce TTS_API_KEY in production | Phase 1a §5.1 ships the `voice_tts` startup guard. Phase 1b honors it as a CONSUMER: `_try_voice_tts` includes `Authorization: Bearer` when the env is set; the production-mode guarantee comes from the `voice_tts` startup guard, not from hub. |
| #9 failure cascade | §2.2 enforces: voice_tts → Voicebox failover → text-only. No cloud TTS. No LLM. Hub greeting still renders text even if both backends fail (browser shows existing `Voice check failed` UX). |
| #11 length-cap edge cases | Phase 1b always uses `trim_for_speech(text, cap=220)`, which Phase 1a §4 guarantees deterministic. |
| #12 tests | §3.8 covers the test list. |

Items NOT addressed by Phase 1b (because they belong elsewhere):

- #2 redirect Phase 1.5 to hub/app.py — resolved by `ADAPTER_DESIGN_PHASE1_5.md`.
- #3 server-side sensitive enforcement — resolved by Phase 1.5 (`rag_core`).
- #4 reject B-prime / adopt Option C — resolved by Phase 1.5.
- #7 reconcile metadata enforcement — resolved by Phase 1.5.
- #8 JAX `/ask` — explicitly NOT pursued in Phase 1b; Phase 1.5 uses Option C instead.
- #10 cache observability — Phase 2.

## 5. Cross-doc isolation guarantees

Phase 1b touches exactly these files:

- `/home/msr8109/projects/hub/app.py` (modify ~80 lines across constants, helper, route handlers, inline JS — see §3).
- `/home/msr8109/projects/hub/tests/test_hub_tts.py` (new file).
- `/home/msr8109/.config/systemd/user/hub.service` (env vars — see §3.7).

Phase 1b does NOT touch:

- `/home/msr8109/projects/cfo_pipeline/dashboard/dashboard.py` (Phase 1a).
- `/home/msr8109/projects/cfo_pipeline/scripts/daily_sync.py` (Phase 1a).
- `/home/msr8109/jax-bot/jax_bot.py` (Phase 1a).
- `/home/msr8109/projects/voice_tts/server.py` (the `TTS_API_KEY` startup guard is Phase 1a).
- `/home/msr8109/projects/qdrant_master_search/master_search_api.py` (Phase 1.5).
- Any future `/home/msr8109/projects/rag_core/` files (Phase 1.5).
- Voicebox container (kept running as failover; Phase 1b does not modify it).

Cross-doc verification once code lands:

- After Phase 1b ships, `grep -n "JARVIS_VOICE_URL\|/api/speak" hub/app.py` should show:
  - `HUB_LEGACY_VOICE_URL` used only inside `_try_legacy_voicebox`.
  - `HUB_TTS_URL` used in `_try_voice_tts`.
  - No call to `voicebox` outside the failover path.
- After Phase 1b ships, `grep -n "voice_tts\|127.0.0.1:8050\|voicebox\|127.0.0.1:8110\|17493" cfo_pipeline/dashboard/dashboard.py` should still show the Phase 1a result (i.e., still no Voicebox call sites; the new Phase 1a `voice_tts` call sites only).

## 6. Risks

- **Voice change for Mike's ear.** Today the greeting plays through the legacy Jarvis Voicebox profile. After Phase 1b, the primary path plays through `voice_tts` (XTTS-v2 with the canonical `mike` voice). Mike must audit the new greeting voice quality and either approve or revert via `HUB_TTS_FAILOVER_ENABLED=1` + a temporary env switch that prefers Voicebox first.
- **Probe latency.** `collect_status()` probes 17 service cards at 450 ms each. `build_greeting()` blocks on `collect_status()`. Phase 1b does not change this — but the cascade `collect_status → build_greeting → trim_for_speech → voice_tts` makes the greeting cold-call latency longer than the Voicebox-only path was. Mitigation: keep `TIMEOUT = 0.45` at `app.py:19`; speak the greeting only when audio is unlocked; warmup `voice_tts` with the canonical greeting on hub startup if Mike approves.
- **Greeting drift.** `build_greeting()` text changes with service health (`{up} of {total} surfaces are online, with {down} needing attention`). Cache hit rate is therefore low. Phase 1b does not normalize the greeting text — that would change visible UX. The cache-warm price is paid on each load with a different `down` count.
- **Audio element reuse.** The same `jarvisBootAudio` element holds the greeting. If the user clicks Replay during a probe-driven greeting change, the cached `pendingBootUrl` may point at a stale greeting. Acceptable in Phase 1b; document for Mike.
- **Phase 1.5 collision.** `synthesize_jarvis_audio()` is shared between greeting (Phase 1b) and Search Chat readback (Phase 1.5). Phase 1.5 may later replace the helper or split it. Phase 1b should NOT make the helper signature unique to greetings — keep `synthesize_jarvis_audio(text)` so Phase 1.5 can override or shadow it without a rename.
- **localStorage policy.** Mike's persistence preference is stored in `localStorage`. If Mike uses a fresh browser profile or clears storage, the greeting will not autoplay until he clicks Start Jarvis once. Same behavior as Phase 1a CFO dashboard.
- **Failover semantic mismatch.** Voicebox `/api/speak` accepts `voice_id="jarvis"`, while `voice_tts` `/speak` uses `voice="mike"`. The two voices are not identical. If Mike's preference is the Voicebox JARVIS profile, then Phase 1b's primary path produces a different voice. This is a feature-not-bug per Phase 1 voicebox-keep-as-failover decision, but it's worth flagging.

## 7. Verification plan

| Test | Setup | Expected |
|---|---|---|
| `/healthz` baseline | Restart hub with new env | `GET /healthz` still returns `{"ok": true, "name": "MASTER DASHBOARD", "read_only": true, "cards": 17}`. |
| No key in browser | Fetch `/` HTML | HTML contains no `TTS_API_KEY` value. |
| Greeting plays via voice_tts | Load `/`, audio unlocked | `<audio id="jarvisBootAudio">` plays; backend logs `[TTS] caller=hub-events status=200 path=voice_tts`. |
| Greeting failover to Voicebox | Stop `voice_tts`, keep Voicebox up | Greeting still plays; backend logs primary failure + `path=voicebox`; response header `X-Hub-TTS-Path: voicebox`. |
| Greeting both-down | Stop voice_tts and Voicebox | Greeting fails; browser shows `Voice check failed`; status text stays visual; no crash. |
| Audio locked across reload | Disable in `localStorage`, reload | Greeting fetches but does not autoplay; `Start Jarvis` button visible. |
| Audio enabled persists | Enable, reload | Greeting autoplays. |
| Phase 1.5 isolation | Hit `/api/search-chat` with fixture query | Same response shape as before Phase 1b. |
| Phase 1a isolation | Run CFO dashboard test suite unchanged | All Phase 1a CFO tests still pass. |
| Truncation enforced | Force `build_greeting()` to return a >220-char string via a probe stub | TTS payload `text` length ≤ 220; verified via mocked `urlopen` capturing the JSON. |
| Search Chat readback still works | Hit `/api/jarvis/speak` with `{text: "hi"}` | 200 audio/wav. |
| `X-Caller` set | Capture outbound voice_tts request in a test | Header includes `X-Caller: hub-events`. |
| Bearer set when key configured | env `TTS_API_KEY=abc`, mock voice_tts | Outbound request has `Authorization: Bearer abc`. |
| `X-Hub-TTS-Path` header set | Hit `/api/jarvis/greeting/speak` | Response includes header. |
| Failover disabled | env `HUB_TTS_FAILOVER_ENABLED=0`, voice_tts down | `/api/jarvis/greeting/speak` returns 502; browser shows visual fallback. |

## 8. Phase 1b decisions Mike pre-approved (Path 2 framing)

- Phase 1b is the Phase 1 work that targets MASTER DASHBOARD.
- Phase 1b reuses Phase 1a's TTS proxy helper conventions and the shared truncation helper.
- Phase 1b keeps Voicebox at 127.0.0.1:8110 as failover (per the original Phase 1 voicebox-keep-as-failover decision).
- Phase 1b uses `X-Caller: hub-events`.
- Phase 1b uses the same `localStorage` unlock pattern that Phase 1a chose for CFO dashboard.
- Phase 1b makes no Search Chat changes — that lane is owned by Phase 1.5.

## 9. Recommended implementation order

1. Land Phase 1a (shared `voice_text_utils` module + `voice_tts` startup guard + JAX adapter + CFO adapter).
2. Verify `voice_tts` is healthy with `TTS_API_KEY` enforced.
3. Land Phase 1b:
   a. Add Phase 1b env vars to `hub.service`.
   b. Add `voice_text_utils` import to `hub/app.py`.
   c. Replace `synthesize_jarvis_audio` with the failover-aware version.
   d. Add `localStorage` unlock helpers and gate `playGreeting()`.
   e. Add `X-Hub-TTS-Path` header.
   f. Add tests.
   g. Restart `hub.service`.
4. Verify greeting plays via voice_tts; verify failover by stopping voice_tts.
5. Two-week stabilization window before Phase 1.5 lands.
6. After Phase 1.5 lands and is stable, run a separate Rule 23 to retire Voicebox.

## 10. Decisions locked for Phase 1b

Mike delegated these decisions to Codex on 2026-05-18 for the revised Rule 23 pass.

1. **Primary voice**: `voice_tts` `mike` is primary. Legacy Voicebox remains failover only during the two-week stabilization window.
2. **Service-card status-transition speaker**: NO for Phase 1b. The hub probe loop runs frequently and status flips would be noisy. Phase 1b speaks the greeting path only.
3. **Mute toggle location**: place it next to `jarvisEnableBtn` / `jarvisReplayBtn` in the existing `jarvisControls` div. Do not move voice controls into the topbar.
4. **localStorage key**: use `hubGreetingAudioEnabled`, parallel to the CFO convention `cfoEventsAudioEnabled`.
5. **Greeting length cap**: enforce at the `trim_for_speech` call site only. Do not change `build_greeting()` text; the visible HTML can remain the full greeting while spoken audio gets the bounded form.
6. **`X-Hub-TTS-Path` visibility**: expose it in the response and log it. During stabilization, the browser can optionally show whether audio came from `voice_tts` or Voicebox failover.
