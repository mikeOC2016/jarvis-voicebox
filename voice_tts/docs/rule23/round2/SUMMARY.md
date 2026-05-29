# Rule 23 Round 2 Triple-Review Summary

Date: 2026-05-18 (round 2 after Phase 1a revision)
Reviewed: ADAPTER_DESIGN_PHASE1A.md (revised), companion docs, CHATTERBOX_INVESTIGATION_2026_05_18.md

## Verdicts

| Reviewer | Status | Verdict | Bytes |
|---|---|---|---|
| codex-ask  | DONE | NEEDS-CHANGES — 7 numbered items | 2588 |
| cursor-agent | DONE | CONDITIONAL — 3 must-fix items | 6997 |
| perplexity-ask | FAILED (banner only, same as round 1) | Not usable | 83 |

2 of 3 review arms produced usable verdicts. Both agree: NEEDS-CHANGES on the same 3 core issues.

## Convergent must-fix items (both reviewers agree)

### Issue 1: TTS_API_KEY auth has a bypass risk
Codex: ENVIRONMENT variable can be misset; production can start unauthenticated.
Cursor: Production mis-set ENVIRONMENT could leave old bypass behavior.
Fix: Default voice_tts to require TTS_API_KEY when binding non-loopback. Allow unauth ONLY with explicit ALLOW_UNAUTH_TTS=1 in dev/test. No ENVIRONMENT ambiguity. Fail-log loudly when key unset regardless of env.

### Issue 2: /api/events/speak accepts raw caller text
Codex: "can become an arbitrary TTS proxy if it accepts raw text."
Cursor: "For sync_all_summary, ignore client text; build canonical strings server-side. For leak_high, constrain provenance to authenticated dashboard action or DB-backed event id, not arbitrary strings."
Fix: Server constructs speech from structured event fields. Accept event_id or structured result, NOT raw text.

### Issue 3: HIGH leak playback delivery is architecturally incomplete
Codex: "HIGH leak narration cannot reach an active browser session without SSE, WebSocket, polling, or a live event bus."
Cursor: "Remove or redesign daily_sync.py -> /api/events/speak so it does NOT imply queued/offline playback; resolve/delete tts_pending ambiguity vs 'no next-visit queue'."
Fix: Either add SSE/polling, OR remove HIGH leak from Phase 1a and ship sync-all only.

## Codex-only must-fix items

4. Adapter must fail-closed if local TTS key env var is empty; never call /speak unauthenticated.
5. Add service-side X-Caller handling in voice_tts (sanitize/allowlist, log caller/status/latency, NEVER log spoken text or bearer tokens).
6. voice_text_utils tests are internally inconsistent: with cap 30, "Sentence one. Sentence two." (both 14 chars) BOTH fit, so expected output cannot be only first sentence. Add explicit tests for "first sentence fits, second would exceed cap" and "cap <= 0".
7. Keep CFO allowlist tight: sync-all summary + verified HIGH leak only. Do not loosen.

## Cursor-only nice-to-haves (not blocking)

- Optional: dedupe HIGH leak narration by event_id to avoid repeat audio.
- Document or enforce "safe default" for voice_tts on PGX (e.g., default ENVIRONMENT=production when binding non-loopback).

## Recommended next steps

1. Apply the 3 convergent fixes (auth default, /api/events/speak contract, HIGH leak delivery) to ADAPTER_DESIGN_PHASE1A.md.
2. For HIGH leak: my recommendation is to REMOVE it from Phase 1a (ship sync-all only). Adds SSE/polling complexity that warrants its own Phase 1a.5 design pass.
3. Apply the 4 codex-only items (adapter fail-closed, X-Caller in voice_tts, voice_text_utils test fix, allowlist tight).
4. Re-run Rule 23 round 3.
5. Implementation ONLY after both reviewers convert to GO.

## Health snapshot

- voice_tts: healthy, voices=["mike"]
- voicebox: healthy (legacy failover)
- jax-bot.service: active
- No source code modified during this review round
- No Docker/Cloudflare/systemd changes
