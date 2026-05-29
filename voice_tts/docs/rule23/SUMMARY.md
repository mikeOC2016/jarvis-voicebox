# Rule 23 Triple-Review Summary

Date: 2026-05-18
Reviewed: ADAPTER_DESIGN_PHASE1.md, ADAPTER_DESIGN_PHASE1_5.md, JAX_CAPABILITY_AUDIT.md

## Verdicts

| Reviewer | Status | Verdict | Bytes |
|---|---|---|---|
| cursor-agent | DONE | CONDITIONAL — 6 must-fix items | 8840 |
| codex-ask | DONE | NEEDS-CHANGES — 7 must-fix items | 3000 |
| perplexity-ask | FAILED twice (banner + 61-byte short) | Not usable | 61 |

2 of 3 review arms produced usable verdicts.

## CRITICAL FINDING (Codex, not caught by Cursor or audit)

Must-fix item 2 from codex:
  "Resolve dashboard target: CFO dashboard and MASTER DASHBOARD are separate apps;
   Phase 1.5 belongs in /home/msr8109/projects/hub/app.py, not CFO dashboard.py."

Verified by reading hub/app.py: contains the MASTER DASHBOARD greeting, ServiceCards
for qdrant-master-search and jarvis-voice, synthesize_jarvis_audio(), and
build_search_chat_answer() — i.e. the actual Search Chat from Mike's screenshot.

The Phase 1 AND Phase 1.5 design docs point Codex at the WRONG dashboard.
Both need a revision pass before any implementation.

## Consolidated must-fix list (union of cursor + codex)

1. Phase 1 dashboard event narration: keep in CFO dashboard.py (CFO event paths only),
   NO Search Chat / RAG changes there
2. Phase 1.5 dashboard target: redirect from cfo_pipeline/dashboard/dashboard.py to
   projects/hub/app.py (the actual MASTER DASHBOARD with Search Chat)
3. Sensitive-collection allowlist: must enforce server-side after retrieval, on every
   /api/search-chat request — browser hints insufficient
4. Reject B-prime as currently written; closest match to Mike's auto-escalate intent is
   Option C (shared controlled RAG library/service), NOT JAX /ask, because privacy
   enforcement must own retrieval+context+escalation in one bounded path
5. Specify deterministic speech truncation: punctuation split + word-bound fallback,
   hard cap 300/220, single shared helper for JAX + hub
6. Enforce TTS_API_KEY on voice_tts in production; no bearer bypass when unset
7. Reconcile sensitive-payload metadata enforcement: cannot claim "uncrossable" without
   server-side denylist on every external LLM/tool route
8. If JAX /ask is still pursued: Bearer JAX_RAG_API_KEY, bind 127.0.0.1, NO Telegram
   token in HTTP path, log no raw Q/A, preserve Rule 27 token isolation
9. Failure cascade: Claude -> Ollama -> templated, NEVER local-sensitive -> Claude
10. Cache: shared LRU is fine for Phase 1, but monitor under heavy JAX/hub competition
11. Length-cap edge cases: deterministic split + fallback word-boundary truncation
12. Tests: no browser key exposure, allowlist rejection, length caps, citation stripping,
    TTS failure fallback, Phase 1/Phase 1.5 route separation

## Recommendation

DO NOT START IMPLEMENTATION.

Return to design phase. Revise both docs:
- Phase 1: confirm CFO dashboard.py is the correct event-narration target (likely yes
  for CFO-side events, but MASTER DASHBOARD events live in hub/app.py)
- Phase 1.5: redirect target to hub/app.py, adopt Option C (shared RAG library) instead
  of Option B-prime per codex/cursor convergent recommendation
- Both: add server-side sensitive-collection enforcement
- Both: add deterministic speech truncation helper as shared module

After revisions, re-run Rule 23. Implementation only after CONDITIONAL converts to GO
on both reviewers.
