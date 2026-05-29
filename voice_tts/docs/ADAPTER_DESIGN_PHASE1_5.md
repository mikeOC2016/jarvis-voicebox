# MASTER DASHBOARD Search Chat — Design Phase 1.5 (Option C: rag_core)

Date: 2026-05-18
Author: Claude (rewrite of Phase 1.5 after Rule 23 NEEDS-CHANGES)
Supersedes: `ADAPTER_DESIGN_PHASE1_5_DRAFT1.md` (renamed in place; original retained for audit trail)
Companion docs: `ADAPTER_DESIGN_PHASE1A.md` (JAX + CFO event narration), `ADAPTER_DESIGN_PHASE1B.md` (MASTER DASHBOARD event narration), `JAX_CAPABILITY_AUDIT.md` (historical investigation; § "Dashboard Search Chat" obsolete — see §0.1)

PGX-HOST: yes. LOCAL-INFERENCE: yes. CLOUD-JUSTIFIED: optional for Mike-approved non-sensitive Search Chat quality/deep-reasoning mode only; never for sensitive collections.

Gate status: DESIGN ONLY. No implementation approved. No service, Docker, Ollama, Anthropic, or voice container changes are part of this document.

## 0. Why this is a rewrite

Rule 23 cursor + codex returned NEEDS-CHANGES on the original Phase 1.5. Two convergent must-fix items drove a full architecture pivot:

- **Wrong target codebase** (cursor §6, codex §2). DRAFT1 was written against the CFO Pipeline dashboard and the MASTER DASHBOARD context was bolted on. Search Chat actually lives in `/home/msr8109/projects/hub/app.py`, not `cfo_pipeline/dashboard/dashboard.py`. This rewrite targets `hub/app.py` from the first line.
- **Reject B-prime; adopt Option C** (cursor §4 / §8, codex §3). DRAFT1 proposed dashboard pre-check then JAX-as-brain (Option B-prime). Both reviewers agreed this fails the "uncrossable sensitive routing" requirement because JAX's `run_claude()` is a full Claude Code agent with filesystem/tool access — it can re-search PGX/L: drive after the dashboard's narrow Top-K precheck. Privacy enforcement must own retrieval + context + escalation in ONE bounded path. The convergent recommendation is a shared controlled library: Option C.

This rewrite ships Option C as a new Python package `rag_core` that hub/app.py imports. JAX integration via the same package is deferred to Phase 2 cleanup.

Rule 23 must-fix items addressed by this rewrite:

- #2 redirect target from CFO `dashboard.py` to `hub/app.py` ✓
- #3 server-side sensitive enforcement on every request, not browser hints ✓ (§4, §6)
- #4 adopt Option C, not B-prime ✓ (§2, §3)
- #7 reconcile metadata enforcement: denylist runs server-side after retrieval ✓ (§6)
- #9 failure cascade Claude → Ollama → templated, never local-sensitive → Claude ✓ (§7)
- #12 tests for sensitive route enforcement, brain selection, citation handling, denylist override attempts ✓ (§10)

Mike-approved DRAFT1 decisions kept intact:

- Top-K = 8 global with per-collection `k=3`.
- Anthropic daily token cap = 100,000.
- No power-user override on sensitive routing.
- Deep reasoning = Opus, manual checkbox only.
- Logging: metadata only (no raw Q/A storage).
- `snippet_chars` backward-compatible param on Qdrant Master Search.
- `voice_text` is the citation-stripped, 220-char-capped speech text.
- Legacy `/api/jarvis/speak` kept as fallback during stabilization (handled by Phase 1b plumbing).

### 0.1 JAX_CAPABILITY_AUDIT.md status

The audit doc was correct about JAX's internal capabilities but its "Dashboard Search Chat" section (§2) and "Recommendation: Option B-prime" (§5) are obsolete:

- §2 described CFO `dashboard.py` instead of `hub/app.py`.
- §5 recommended B-prime which Rule 23 rejected.

The audit's JAX-side findings (§1.1-1.5) remain valid and are referenced in §3.4 of this doc when justifying why JAX is NOT the brain in Phase 1.5.

## 1. Current state — `hub/app.py` Search Chat path

`hub/app.py` is 401 lines, port 8101, serves the MASTER DASHBOARD seen in Mike's screenshot. Search Chat lives entirely in this file.

Service unit:

```ini
/home/msr8109/.config/systemd/user/hub.service
WorkingDirectory=/home/msr8109/projects/hub
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8101
```

App constants (`hub/app.py:17-24`):

```python
JARVIS_VOICE_URL = "http://127.0.0.1:8110"
QDRANT_SEARCH_URL = "http://127.0.0.1:8100/search"
DQ_SALES_URL = "http://127.0.0.1:8090/api/dq/sales"
DEFAULT_AUTO_SEARCH_QUERY = "SALES TODAY FOR DQS"
```

(After Phase 1b lands, `JARVIS_VOICE_URL` is renamed to `HUB_LEGACY_VOICE_URL` per `ADAPTER_DESIGN_PHASE1B.md` §3.2.)

### 1.1 Current frontend Search Chat

Inline HTML in `render_dashboard()` at `app.py:300-302` builds:

- `<form class="searchForm" id="searchChatForm">` with `<input id="searchChatInput">` and an `Ask` button.
- `<div class="searchAnswer" id="searchChatAnswer">` for the rendered answer.
- `<ul class="searchResults" id="searchChatResults">` for source rows.
- `<audio class="voiceAudio" id="jarvisSearchAudio" preload="none" playsinline controls>` for voice playback.

Inline JS:

- `runSearch(q)` at `app.py:320` POSTs `{q}` to `/api/search-chat`.
- On success: renders `p.answer`, fills `searchResults` from `p.results.map(item => ...)`, then calls `fetchVoice('/api/jarvis/speak', {text: p.answer})` and plays the returned WAV.
- On Chrome autoplay block: keeps WAV loaded; status text says `Voice is ready. Click play if Chrome blocked autoplay.`.
- `searchChatForm` submit handler at `app.py:321` calls `runSearch(...)` with the trimmed input.
- Page load auto-fires `runSearch(DEFAULT_AUTO_SEARCH_QUERY)` 1300 ms after load (`app.py:322`).

### 1.2 Current backend Search Chat

`/api/search-chat` at `app.py:383-396`:

```python
@app.post("/api/search-chat")
async def api_search_chat(request: Request) -> JSONResponse:
    payload = await request.json()
    query = str(payload.get("q") or payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    try:
        if _is_dq_sales_today_query(query):
            answer = build_dq_sales_answer(query, fetch_dq_sales())
        else:
            answer = build_search_chat_answer(query, search_qdrant_master(query))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="PGX search unavailable") from exc
    return JSONResponse({"ok": True, "speakable": True, **answer})
```

Helpers:

- `_is_dq_sales_today_query(query)` at `app.py:209-212` — shortcut for "SALES TODAY FOR DQS" style queries.
- `build_dq_sales_answer(query, fetch_dq_sales())` at `app.py:215-255` — bypasses Qdrant entirely, calls `http://127.0.0.1:8090/api/dq/sales`.
- `search_qdrant_master(query)` at `app.py:161-164` — calls `http://127.0.0.1:8100/search?q=...&k=3&total=6`.
- `build_search_chat_answer(query, search_payload)` at `app.py:258-279` — non-LLM templated summary:
  ```text
  I searched PGX for {query}. I found {total_hits} matches across {collections} collections. Top result from {top_collection}: {top_snippet}
  ```

There is no LLM synthesis today. The audio readback speaks this templated sentence.

### 1.3 Qdrant Master Search service

Service file: `/home/msr8109/projects/qdrant_master_search/master_search_api.py`. Service unit: `/home/msr8109/.config/systemd/user/qdrant-master-search.service`.

Confirmed behavior (DRAFT1 §1 verified):

- Default Qdrant URL `http://localhost:6333`, embedding model `all-MiniLM-L6-v2`, dim 384.
- `/search` embeds query once, fans out to all 384-dim collections in parallel, sorts globally by score, returns top `total` hits with `payload_keys` array.
- Current snippet cap ~280 chars.

### 1.4 What changes in Phase 1.5

The change is NOT to rewrite the Search Chat path inside `hub/app.py`. The change is:

- Pull retrieval + privacy + LLM routing + answer synthesis into a new `rag_core` package.
- Make `api_search_chat` a thin caller: parse request, call `rag_core.rag_answer(query, allow_external)`, render the result.
- Keep the DQ sales shortcut where it is (it bypasses Qdrant; `rag_core` does NOT subsume it).
- Keep the Search Chat audio readback wiring; only the text source changes (templated → grounded answer).

## 2. Architecture — Option C (shared rag_core)

```text
+------------------------------------------------+
|  Browser MASTER DASHBOARD (hub/app.py render)  |
|  - Search Chat form, brain toggle (default Local) |
|  - Renders answer + sources + voice readback   |
+--------------------+---------------------------+
                     | POST /api/search-chat
                     | {q, allow_external}
                     v
+------------------------------------------------+
|  hub/app.py /api/search-chat                   |
|  - Thin wrapper: parse + call + format         |
|  - NEVER touches Qdrant, Ollama, Claude, or    |
|    privacy logic directly                      |
+--------------------+---------------------------+
                     | rag_core.rag_answer(q, allow_external)
                     v
+================================================+
|  rag_core (new Python package, separate file)  |
|  PUBLIC: rag_answer(query, allow_external)     |
|                                                |
|  Step 1: Retrieve via Qdrant Master Search     |
|          (HTTP to 127.0.0.1:8100/search)       |
|                                                |
|  Step 2: SERVER-SIDE sensitive denylist check  |
|          on every retrieved hit                |
|          - any sensitive hit -> force local    |
|          - allow_external is IGNORED if any    |
|            sensitive hit is in Top-K           |
|                                                |
|  Step 3: Brain selection                        |
|          - If forced local: brain=local        |
|          - Else if allow_external: brain=claude|
|          - Else: brain=local                   |
|                                                |
|  Step 4: Prompt build (system + grounding)     |
|                                                |
|  Step 5: LLM call                              |
|          - local: Ollama gemma3:27b or         |
|            llama3.3:70b (context-size auto)    |
|          - claude: Anthropic API               |
|            sonnet-4-6 default,                 |
|            opus-4-7 if deep_reasoning          |
|                                                |
|  Step 6: Citation validation + voice_text gen  |
|          - strip citations -> voice text       |
|          - trim_for_speech(voice_text, 220)    |
|                                                |
|  Returns: {                                    |
|    answer, voice_text, sources,                |
|    brain_used, model_used,                     |
|    blocked_for_privacy, blocked_collections,   |
|    fallback_used                               |
|  }                                             |
+================================================+
                     |
                     v
+------------------------------------------------+
|  hub/app.py /api/search-chat returns JSON      |
|  Browser renders answer + sources + plays      |
|  voice via existing /api/jarvis/speak (Phase 1b|
|  voice_tts path)                               |
+------------------------------------------------+
```

Reasoning behind Option C vs B-prime (resolves Rule 23 #4):

- B-prime put the privacy decision in `hub/app.py` and the answer generation in JAX `/ask`. JAX's brain is `claude -p --dangerously-skip-permissions --max-turns 20` — a full agent. Even if hub pre-checked sensitive collections, JAX could re-search PGX/L: drive on its own, bypassing the precheck. Cursor §7 and codex §3 both flagged this. Privacy enforcement must own retrieval + context + escalation in one bounded path.
- Option C puts retrieval + privacy + answer generation in `rag_core`. The dashboard cannot bypass it because the dashboard does not have a different code path to Qdrant or to any LLM. JAX cannot bypass it because the dashboard does not call JAX in Phase 1.5.
- The only way to talk to a cloud LLM from `hub/app.py` after Phase 1.5 is through `rag_core.rag_answer()`. Mike's compliance posture relies on that being the only door.

JAX-as-brain is explicitly NOT pursued in Phase 1.5. JAX may later import `rag_core` for consistency (Phase 2 cleanup), but `hub/app.py` does NOT call JAX over HTTP.

## 3. `rag_core` package design

### 3.1 Location and layout

```text
/home/msr8109/projects/rag_core/
  pyproject.toml          (optional; package can also be loaded via PYTHONPATH)
  rag_core/
    __init__.py           (exports rag_answer)
    answer.py             (rag_answer orchestrator)
    retrieval.py          (Qdrant Master Search HTTP client)
    privacy.py            (sensitive denylist + per-hit classification)
    prompts.py            (system prompts + context template)
    backends/
      __init__.py
      ollama.py           (local LLM client; gemma3:27b, llama3.3:70b)
      anthropic.py        (Claude API client; sonnet-4-6, opus-4-7)
      templated.py        (no-LLM fallback summary; current templated sentence)
    tests/
      test_privacy.py
      test_retrieval.py
      test_answer.py
      test_backends_ollama.py
      test_backends_anthropic.py
      test_backends_templated.py
      test_integration_sensitive_forced_local.py
      test_integration_brain_selection.py
      test_integration_citation_stripping.py
      test_integration_denylist_override_attempts.py
```

Phase 1.5 vendors a thin copy of `voice_text_utils.trim_for_speech` (Phase 1a §4) into `rag_core/voice_text_utils.py`, OR — preferred — imports it from the standalone `voice_text_utils` package shipped in Phase 1a.

### 3.2 Public API

```python
# rag_core/__init__.py
from .answer import rag_answer

__all__ = ["rag_answer"]
```

```python
# rag_core/answer.py

from typing import TypedDict, Literal

class RagSource(TypedDict):
    source_id: str        # "collection:id"
    collection: str
    id: str
    score: float
    snippet: str

class RagAnswer(TypedDict):
    ok: bool
    query: str
    answer: str           # full answer with inline [collection:id] citations
    voice_text: str       # citation-stripped, <=220 chars
    sources: list[RagSource]
    brain_requested: Literal["local", "claude"]
    brain_used: Literal["local", "claude", "templated"]
    model_used: str       # e.g. "gemma3:27b", "claude-sonnet-4-6", "templated"
    blocked_for_privacy: bool          # True if force-local fired
    blocked_collections: list[str]      # collections that triggered force-local
    fallback_used: Literal["none", "local_to_templated", "claude_to_local", "claude_to_templated"]
    total_hits: int
    collections_searched: int
    banner: str | None                  # user-facing reason string, e.g. "Routed locally for privacy."

def rag_answer(
    query: str,
    *,
    allow_external: bool = False,
    deep_reasoning: bool = False,
    top_total: int = 8,
    per_collection_k: int = 3,
    snippet_chars: int = 900,
) -> RagAnswer:
    """
    Single bounded path for MASTER DASHBOARD Search Chat answer generation.

    Privacy contract:
      - If ANY retrieved hit's collection or inferred class is sensitive,
        brain is forced to local regardless of allow_external.
      - allow_external=True does NOT override sensitive routing. Ever.
      - There is no power-user override.

    Failure cascade:
      - Claude -> local -> templated  (when allowed_external True and not blocked)
      - Local  -> templated           (when forced local or allow_external False)
      - Never local-sensitive -> Claude.
    """
```

### 3.3 Why a function, not a service

A library import has three properties relevant to Rule 23 §4:

1. The privacy decision is in the same process as the dashboard. A new HTTP service would be a separate trust boundary; if its auth ever weakened, sensitive content could egress.
2. There is no HTTP listener for an attacker (or a misconfigured JAX) to call. The only caller is `hub/app.py`. Phase 2 may add JAX as a second caller via `import rag_core`, not via HTTP.
3. The 30 s end-to-end latency budget is preserved; no extra TCP hop.

If Phase 2 demands a service (e.g., Windows tray, browser extension), that service must wrap `rag_core.rag_answer` and bind to `127.0.0.1` with a separate `RAG_API_KEY`. Phase 1.5 does NOT ship that service.

### 3.4 Why not JAX `/ask`

Per JAX_CAPABILITY_AUDIT.md §1.1, JAX's brain is:

```text
claude -p <prompt> --dangerously-skip-permissions --max-turns 20
```

That subprocess can read files, run bash, query Qdrant directly, search L: drive — anything Claude Code can do. Even if the dashboard restricts which queries reach JAX, JAX itself can fetch sensitive context that the dashboard intended to block. Cursor §7 ("privacy enforcement must own retrieval+context+escalation in one bounded path") and codex §3 ("Best match to Mike's intent is Option C") both flagged this.

Phase 1.5 deliberately does not add `POST /ask` to JAX. The Rule 27 token isolation argument from DRAFT1 §4 is preserved (no new HTTP endpoint on jax-bot), and the privacy argument from Rule 23 is also preserved.

If Phase 2 ever pursues a JAX/`rag_core` reuse story, it imports `rag_core` from inside `jax_bot.py` and calls `rag_answer()` directly. No HTTP endpoint. No `JAX_RAG_API_KEY`. No multi-process privacy enforcement.

## 4. UI changes in `hub/app.py`

### 4.1 Brain toggle

The DRAFT1 `Local` / `Claude` segmented control is replaced with a simpler boolean per Mike's "default local, opt-in cloud" framing:

- One small checkbox to the right of the `Ask` button: `Allow external (Claude)`.
- Default state: **unchecked** (i.e., `allow_external=false`).
- When unchecked, the dashboard sends `allow_external: false` to `/api/search-chat`.
- When checked, the dashboard sends `allow_external: true`. The `rag_core` privacy check still runs server-side; the box is a request, not a guarantee.
- An optional sub-control appears when checked: a smaller checkbox `Deep reasoning (Opus)`. Default unchecked.

Why a single boolean instead of segmented control: simpler markup; same semantic. Mike's privacy posture is "default local, allow cloud sometimes". A segmented `Local/Claude` invites the misread that Claude is uncrossable; in fact local is uncrossable.

### 4.2 Persistence

```text
localStorage.hubAllowExternal = "0" | "1"
localStorage.hubDeepReasoning = "0" | "1"
```

Default to `"0"`. Reset to `"0"` on any 500 from `/api/search-chat`.

### 4.3 Banner rendering

If `rag_core` returns `blocked_for_privacy=true`, the dashboard renders a small chip above the answer:

```text
Routed locally for privacy: <list of blocked collection names>
```

If `fallback_used != "none"`, render:

```text
Claude unavailable; routed locally.
```

or

```text
Local LLM unavailable; using templated summary.
```

depending on the fallback value.

If neither, no banner.

### 4.4 Request and response shape

Browser → hub:

```json
{ "q": "...", "allow_external": false, "deep_reasoning": false }
```

hub `/api/search-chat` → browser:

```json
{
  "ok": true,
  "speakable": true,
  "query": "...",
  "answer": "Grounded answer with [collection:id] citations.",
  "voice_text": "Spoken text without citations, <=220 chars.",
  "sources": [
    {"source_id":"ldrive_docs:abc","collection":"ldrive_docs","id":"abc","score":0.77,"snippet":"..."}
  ],
  "brain_requested": "local",
  "brain_used": "local",
  "model_used": "gemma3:27b",
  "blocked_for_privacy": false,
  "blocked_collections": [],
  "fallback_used": "none",
  "total_hits": 8,
  "collections_searched": 33,
  "banner": null
}
```

This is the same shape `rag_core.rag_answer` returns plus the `ok` and `speakable` flags. The DQ sales shortcut at `app.py:215-255` returns a separate (existing) shape; the dashboard renders both shapes from the same form handler.

### 4.5 Voice readback (reuse Phase 1b plumbing)

Per Phase 1b §3.3, `synthesize_jarvis_audio()` was migrated to `voice_tts` (primary) with Voicebox failover. Phase 1.5 reuses that exact helper without re-implementation:

- Browser receives `{answer, voice_text, ...}`.
- Browser POSTs `voice_text` to `/api/jarvis/speak` (existing route at `app.py:362-380`, now backed by Phase 1b's failover-aware `synthesize_jarvis_audio`).
- `rag_core` is responsible for `voice_text` being citation-free and ≤220 chars (via the shared `trim_for_speech` helper with `strip_citations=True`).
- `X-Caller` on the outbound `voice_tts` call stays `hub-events` (set by Phase 1b inside `synthesize_jarvis_audio`). Phase 1.5 does not change that header; logging can still distinguish greeting vs search-chat readback by event metadata at the endpoint level if needed.

If Mike wants Search Chat readback under its own `X-Caller: dashboard-rag`, then Phase 1.5 adds a thin wrapper around `synthesize_jarvis_audio` that overrides the header for `/api/jarvis/speak` only. Phase 1.5 draft prefers reusing `hub-events` to minimize Phase 1b churn; flag in §11 Q1.

## 5. Retrieval (kept from DRAFT1, condensed)

`rag_core.retrieval` calls the existing Qdrant Master Search HTTP API. No changes to embedding model, candidate selection, or scoring.

Default request from `rag_core`:

```text
GET http://127.0.0.1:8100/search?q=<query>&k=3&total=8&snippet_chars=900
```

`snippet_chars` is a new backward-compatible query parameter added to Qdrant Master Search (`master_search_api.py`). Default behavior preserved at ~280 chars when the parameter is omitted. `rag_core` requests 900 chars for synthesis quality. DRAFT1 §5 covered this; unchanged.

Phase 1.5 does NOT add reranking, query rewriting, recency boost, collection priority, or re-embedding. Phase 2.

## 6. Privacy posture (resolves Rule 23 #3, #4, #7)

### 6.1 Where enforcement runs

Privacy enforcement runs inside `rag_core.privacy.classify(sources)` AFTER retrieval and BEFORE any prompt build or LLM call. It runs on every call to `rag_answer`. There is no caller-supplied bypass. The browser CANNOT influence this decision beyond `allow_external` (a request, not a guarantee).

### 6.2 Denylist sources

The denylist is configured server-side in three layers:

1. **Hard-coded collection denylist** in `rag_core/privacy.py`. Initial list (Mike pre-approved in DRAFT1 §8):

   ```text
   mike_files
   ldrive_docs
   ldrive_master_file
   bank_statements_2026
   sales_reports
   chatvault_chunks
   ai_brain_unified
   apex_paper_trades
   apex_trade_analysis
   apex_trade_theses
   apex_signals
   apex_ml_predictions
   apex_backtests
   apex_knowledge
   apex_self_learning
   self_learning_scores
   apex_portfolio_optimization
   ```

2. **Metadata classifier**. For each retrieved hit, if `payload_keys` contains tokens indicating personal/legal/financial/medical content (e.g., `ssn`, `passport`, `account_number`, `routing_number`, `tax_id`, `medical_record`, `health`, `attorney_client`), force local. Until Qdrant Master Search exposes richer metadata, this classifier uses `payload_keys` only — that is acknowledged as a partial enforcement (Rule 23 cursor §3) and §6.5 below describes the mitigation.

3. **Source path heuristic**. If `payload_keys` includes a `source` or `source_path` field hint that is L: drive (`/mnt/bookkeeper/`, `\\MIKE-OFFICE-ACC\Desktop`, or any L: drive marker), force local.

If layer 1, 2, or 3 fires for ANY hit in the Top-K, the whole query is forced local. There is no per-hit redaction in Phase 1.5; the whole query routes local.

### 6.3 Override prevention

- `allow_external=True` does NOT override force-local. `rag_core` ignores `allow_external` when any sensitive hit is present.
- The dashboard cannot pass a `bypass_privacy=True` flag. The function signature does not accept one.
- A future caller (e.g., JAX in Phase 2) imports the same module and is bound to the same rule.
- Browser `localStorage` cannot influence the server-side decision.

### 6.4 Candidate Claude-safe collections (informational; Mike review)

DRAFT1 §8 listed these. Phase 1.5 keeps them as the "not on denylist" set rather than an explicit allowlist, so that newly-added Qdrant collections default to non-sensitive only when they pass the metadata classifier:

```text
apex_seeking_alpha, apex_sentiment, apex_quant_math, apex_reddit, apex_reddit_sentiment,
apex_fred_data, apex_macro_fred, apex_google_trends, apex_dark_pool, apex_congress_trades,
apex_congress, apex_sec_filings, apex_whale_alerts, apex_job_trends, apex_alt_economy,
apex_macro, market_regimes, apex_multiasset
```

Any new collection NOT in the denylist still passes through the metadata classifier and path heuristic. If either trips, force local.

### 6.5 Metadata gap mitigation (resolves Rule 23 #7)

Cursor §3 / §7 flagged that "uncrossable" cannot be claimed while Master Search exposes only `payload_keys`. Phase 1.5 mitigates by:

- Treating the metadata classifier as **best-effort defense in depth**, not the primary mechanism. The hard-coded collection denylist is the primary mechanism.
- Documenting that any collection not on the denylist that contains sensitive content is a configuration bug to add to the denylist. The hard-coded list is the contract.
- Adding a Phase 2 task to enrich Master Search responses with `payload_class` (hashed tag class summary) so the classifier becomes more reliable. Phase 1.5 does not change Master Search beyond the `snippet_chars` parameter.

This is the same posture cursor §3 called "aspirational vs proven" — Phase 1.5 explicitly downgrades the marketing claim. The privacy posture is "collections in the denylist are uncrossable; collections not in the denylist are best-effort". Mike acknowledges this in §11 Q3.

### 6.6 Tests for denylist override attempts (Rule 23 #12)

`rag_core/tests/test_integration_denylist_override_attempts.py` covers:

- Caller sets `allow_external=True` with a sensitive-collection hit → `brain_used="local"`, `blocked_for_privacy=True`.
- Caller passes a forged `bypass_privacy=True` kwarg → `TypeError` because the signature does not accept it.
- Multiple sensitive hits → `blocked_collections` lists all of them.
- One sensitive + many safe hits → still forced local.
- All safe hits and `allow_external=True` → `brain_used="claude"`.
- All safe hits and `allow_external=False` → `brain_used="local"`.
- Empty retrieval → `brain_used="templated"` with `"I don't have enough information"` answer; no LLM call.

## 7. Failure cascade (resolves Rule 23 #9)

Inside `rag_core.rag_answer`:

| Path | Trigger | Fallback |
|---|---|---|
| Claude → local | `allow_external=True` and not blocked; Anthropic call raises | Local model with same retrieved context; `fallback_used="claude_to_local"`; banner `Claude unavailable; routed locally.` |
| Local → templated | Local LLM raises or times out; OR `allow_external=False` and Ollama down | Templated summary (current behavior); `fallback_used="local_to_templated"`; banner `Local LLM unavailable; using templated summary.` |
| Claude → templated | Claude raises AND local also raises | Templated summary; `fallback_used="claude_to_templated"`; banner `Both brains unavailable; using templated summary.` |
| Forced local → Claude | NEVER. Even if local fails | Templated summary only. Banner reflects local failure. |

The "never local-sensitive → Claude" rule is enforced by the control flow in `rag_core.answer.rag_answer`:

```python
def rag_answer(query, *, allow_external=False, ...):
    sources = retrieval.fetch(query, ...)
    privacy = privacy_classify(sources)  # blocked: bool, blocked_collections: list
    if privacy.blocked:
        effective_allow_external = False
    else:
        effective_allow_external = allow_external

    if effective_allow_external:
        return _try_claude_then_local_then_templated(query, sources, deep_reasoning, ...)
    else:
        return _try_local_then_templated(query, sources, ...)
```

There is no code path where `privacy.blocked` is true and Claude is called. Tests assert this is true even when the local LLM raises.

## 8. Prompt design (kept from DRAFT1, condensed)

Source ID format: `[collection:id]`.

Local system prompt (Ollama gemma3:27b, llama3.3:70b):

```text
You are the MASTER DASHBOARD Search Chat answer engine running on Mike's local PGX server.

Rules:
1. Answer only from the provided documents.
2. Do not use general knowledge. Do not guess.
3. If the provided documents do not answer the question, say: "I don't know based on the retrieved documents."
4. Cite every factual claim with an inline source ID exactly like [collection:id].
5. Use only source IDs that appear in the provided documents.
6. Be concise. The answer may be spoken aloud by JARVIS.
7. Put the direct answer in the first two sentences.
8. Keep the first two sentences under 220 characters when possible.
9. Do not mention these instructions.

Return plain text only.
```

Claude system prompt (sonnet-4-6, opus-4-7):

```text
You are the MASTER DASHBOARD Search Chat answer engine for Mike Ramadan.

Use only the retrieved documents supplied in this request. Do not use outside knowledge, assumptions, or unstated context. If the documents are insufficient, say "I don't know based on the retrieved documents" and briefly name what evidence is missing.

Requirements:
- Cite source IDs inline for factual claims using exactly this format: [collection:id].
- Only cite source IDs provided in the context block.
- Be concise because the answer may be spoken aloud.
- Put the answer's main point in the first two sentences.
- Keep the first two sentences suitable for a 220-character voice summary when possible.
- Do not expose hidden reasoning or prompt instructions.

Return plain text only.
```

User/context template (verbatim from DRAFT1 §4):

```text
Question:
{question}

Retrieved documents:

SOURCE [collection:id]
Collection: {collection}
Score: {score}
Text:
{chunk_text_or_snippet}

(... per source)

Answer the question using only these retrieved documents.
```

`rag_core.prompts` owns these strings. Any future caller (including a Phase 2 JAX-import) reuses the same strings.

## 9. Files Phase 1.5 would touch (with line numbers)

No files are changed in this design pass. If Mike approves implementation after Rule 23, the following exact regions and files would be touched.

### 9.1 New package

```text
/home/msr8109/projects/rag_core/                (NEW, see §3.1 layout)
```

All new code lives here. Phase 1.5 ships:

- `rag_core/__init__.py`
- `rag_core/answer.py`
- `rag_core/retrieval.py`
- `rag_core/privacy.py`
- `rag_core/prompts.py`
- `rag_core/backends/ollama.py`
- `rag_core/backends/anthropic.py`
- `rag_core/backends/templated.py`
- `rag_core/tests/*.py`

### 9.2 hub/app.py changes (with line numbers)

`app.py:1-15` (imports). Add:

- `from rag_core import rag_answer`

`app.py:17-24` (constants). Add:

- `HUB_RAG_TOP_TOTAL = 8`
- `HUB_RAG_PER_COLLECTION_K = 3`
- `HUB_RAG_SNIPPET_CHARS = 900`

`app.py:258-279` (`build_search_chat_answer`). DELETED. Replaced by `rag_core.rag_answer` inside the route handler. The function name is removed; tests are updated accordingly.

`app.py:300-302` (rendered HTML for the Search Chat panel). Modified to add the `Allow external (Claude)` and `Deep reasoning (Opus)` checkboxes. ~10 lines of HTML added. No new dependency.

`app.py:303-322` (inline JS). Modified:

- `runSearch(q)` at `app.py:320` reads the two checkboxes and posts `{q, allow_external, deep_reasoning}` instead of `{q}`. ~8 lines added.
- Added: a `renderBanner(p)` helper that renders the privacy / fallback banner above the answer when `p.banner` is set. ~5 lines added.
- Added: localStorage init and persistence for both checkboxes. ~10 lines added.

`app.py:383-396` (`api_search_chat`). REPLACED with:

```python
@app.post("/api/search-chat")
async def api_search_chat(request: Request) -> JSONResponse:
    payload = await request.json()
    query = str(payload.get("q") or payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    allow_external = bool(payload.get("allow_external") or False)
    deep_reasoning = bool(payload.get("deep_reasoning") or False)
    try:
        if _is_dq_sales_today_query(query):
            answer = build_dq_sales_answer(query, fetch_dq_sales())
            return JSONResponse({"ok": True, "speakable": True, **answer})
        result = rag_answer(
            query,
            allow_external=allow_external,
            deep_reasoning=deep_reasoning,
            top_total=HUB_RAG_TOP_TOTAL,
            per_collection_k=HUB_RAG_PER_COLLECTION_K,
            snippet_chars=HUB_RAG_SNIPPET_CHARS,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="PGX search unavailable") from exc
    return JSONResponse({"ok": True, "speakable": True, **result})
```

`app.py:161-164` (`search_qdrant_master`). DELETED — moved into `rag_core/retrieval.py`. Tests updated.

`app.py:258-279` (`build_search_chat_answer`). DELETED — its templated behavior moves into `rag_core/backends/templated.py`.

`app.py:215-255` (`build_dq_sales_answer`) — UNCHANGED. The DQ sales shortcut stays in `hub/app.py`.

`app.py:209-212` (`_is_dq_sales_today_query`) — UNCHANGED.

`app.py:167-169` (`fetch_dq_sales`) — UNCHANGED.

`app.py:172-176` (`_compact_snippet`) — DELETED only if no remaining caller. (DQ sales path uses it; check before delete.)

### 9.3 Qdrant Master Search change

```text
/home/msr8109/projects/qdrant_master_search/master_search_api.py
```

Add backward-compatible `snippet_chars` query parameter to `/search`. Default behavior at ~280 chars when omitted. `rag_core` requests 900. Other callers unchanged.

### 9.4 hub service env

```text
/home/msr8109/.config/systemd/user/hub.service
```

Add:

```ini
Environment="ANTHROPIC_API_KEY=<server-only>"
Environment="DASHBOARD_RAG_ANTHROPIC_DAILY_TOKEN_CAP=100000"
Environment="DASHBOARD_RAG_DEFAULT_ALLOW_EXTERNAL=0"
```

`TTS_API_KEY` already added by Phase 1b. Confirm both phases coordinate on the env file.

### 9.5 Files explicitly NOT touched

- `/home/msr8109/projects/cfo_pipeline/dashboard/dashboard.py` — Phase 1a.
- `/home/msr8109/projects/cfo_pipeline/scripts/daily_sync.py` — Phase 1a.
- `/home/msr8109/jax-bot/jax_bot.py` — JAX. Phase 1.5 does NOT add `/ask`. Phase 2 may import `rag_core` from inside JAX.
- `/home/msr8109/projects/voice_tts/server.py` — Phase 1a startup guard; Phase 1.5 does not touch.
- `/home/msr8109/projects/hub/app.py` Phase 1b plumbing (`synthesize_jarvis_audio`, route handlers `/api/jarvis/greeting/speak` and `/api/jarvis/speak`) — Phase 1.5 reuses; does not modify.
- Voicebox container — untouched.

## 10. Verification plan (resolves Rule 23 #12)

### 10.1 `rag_core` unit tests

Files in `rag_core/tests/`:

- `test_privacy.py`:
  - Denylist hit forces local even when `allow_external=True`.
  - Metadata classifier sensitive hint forces local.
  - L: drive path hint forces local.
  - Multiple sensitive hits accumulate in `blocked_collections`.
  - Empty Top-K returns `blocked_for_privacy=False`.

- `test_retrieval.py`:
  - Calls Qdrant Master Search at `127.0.0.1:8100/search` with correct query params.
  - Handles 0 hits.
  - Handles transient 5xx (single retry, surfaces failure).

- `test_answer.py`:
  - End-to-end pure mock; assert request shape into each backend.
  - Voice text is citation-stripped and ≤220 chars.

- `test_backends_ollama.py`:
  - Calls `127.0.0.1:11434/api/generate` with correct model name.
  - Auto-selects `llama3.3:70b` for large context.
  - Times out cleanly.

- `test_backends_anthropic.py`:
  - Calls Anthropic Messages API with `claude-sonnet-4-6` default.
  - Switches to `claude-opus-4-7` when `deep_reasoning=true`.
  - Honors daily token cap.
  - Never called when forced local.

- `test_backends_templated.py`:
  - Templated output matches existing `build_search_chat_answer` sentence shape.
  - Used when both LLMs fail.

- `test_integration_sensitive_forced_local.py`:
  - Sensitive hit + `allow_external=True` → no Anthropic call.
  - Sensitive hit + Ollama down → templated summary, NOT Claude.

- `test_integration_brain_selection.py`:
  - Safe + `allow_external=True` + deep_reasoning=true → Opus.
  - Safe + `allow_external=True` + deep_reasoning=false → Sonnet.
  - Safe + `allow_external=False` → Gemma/Llama.

- `test_integration_citation_stripping.py`:
  - Answer with `[col:id]` → voice_text has no brackets.
  - Voice text ≤220 chars after stripping.

- `test_integration_denylist_override_attempts.py`:
  - All attempts in §6.6.

### 10.2 hub/app.py integration tests

`/home/msr8109/projects/hub/tests/test_search_chat_phase1_5.py`:

- DQ sales shortcut still works (bypasses Qdrant and `rag_core`).
- `/api/search-chat` returns the expanded JSON shape.
- Banner text rendered when `blocked_for_privacy=True`.
- Browser HTML contains no API key (`TTS_API_KEY` nor `ANTHROPIC_API_KEY`).
- `localStorage` defaults respected on first load.
- `runSearch` posts `allow_external` and `deep_reasoning` correctly.
- Voice readback via `/api/jarvis/speak` still works (Phase 1b plumbing).
- Phase 1b greeting still works (regression).
- Phase 1a JAX and CFO event narration still work (regression at the file-mtime level; no cross-file effect).

### 10.3 Manual verification (post-implementation)

| Test | Setup | Expected |
|---|---|---|
| Default local | Page load, ask any question | `brain_used=local`, no Anthropic call, ≤3s. |
| Allow external + safe | Check Claude box, ask non-sensitive question | `brain_used=claude`, `model_used=claude-sonnet-4-6`. |
| Allow external + sensitive | Check Claude box, ask about `mike_files` content | `brain_used=local`, `blocked_for_privacy=true`, banner shown. |
| Deep reasoning | Check both Claude and Deep reasoning, safe query | `model_used=claude-opus-4-7`. |
| Anthropic down | Stop Anthropic egress, ask with Claude allowed + safe | Falls back to local; banner shown. |
| Ollama down | Stop Ollama, ask with Claude disallowed | Falls back to templated; banner shown. |
| Both LLMs down | Both off | Templated summary; banner. |
| Token cap | Force daily counter past 100k, allow Claude + safe | Forced local; banner. |
| DQ sales bypass | Ask "SALES TODAY FOR DQS" | DQ shortcut returns; no `rag_core` call; voice readback works. |
| Long answer | Ask question that yields >220 char answer | Full answer rendered; `voice_text` ≤220 chars; no `play full` button. |
| Citation stripping | Returned answer contains `[col:id]` | Voice text strips brackets. |
| Voice down | Stop voice_tts and Voicebox both | Text answer renders; voice failure visible; `/api/search-chat` still returns 200. |

## 11. Decisions locked for Phase 1.5

Mike delegated these decisions to Codex on 2026-05-18 for the revised Rule 23 pass.

1. **Voice readback `X-Caller`**: use a separate `X-Caller: dashboard-rag` for Search Chat readback. Phase 1b keeps `X-Caller: hub-events`; Phase 1.5 adds a thin wrapper so observability can distinguish event narration from RAG narration.
2. **`rag_core` location**: keep `/home/msr8109/projects/rag_core/` as a standalone sibling package. Phase 2 may import it from JAX, so it should not live inside the hub project.
3. **Denylist contract**: do not allow cloud egress on uncertainty. If any retrieved hit is in a hard-deny collection, trips metadata/path sensitivity, or lacks enough metadata for a safe classification, `rag_core` forces local/templated handling. Search Chat itself remains available; only external LLM routing is blocked.
4. **Anthropic egress default**: hard-code the server-side default to `False`. Do not honor an env var that can default cloud routing on. A future env var may disable external egress entirely, but cannot make cloud the default.
5. **Templated fallback wording**: preserve the existing answer sentence verbatim and add a structured banner/metadata field, not a natural-language `(no LLM available)` suffix inside the answer text.
6. **DQ sales shortcut location**: leave it in `hub/app.py` for Phase 1.5. It has a different data shape and should not expand `rag_core` scope.
7. **JAX `rag_core` import in Phase 2**: keep JAX's Claude Code agent as a separate power mode. Phase 2 may route RAG/search answers through `rag_core`, but should not delete `run_claude` or collapse all JAX behavior into `rag_core` without its own design/review.
8. **Search Chat readback voice**: use `voice_tts` `mike` for Search Chat readback. Legacy Voicebox remains failover only.
