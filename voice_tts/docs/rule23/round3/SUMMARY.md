# Rule 23 Round 3 Triple-Review Summary

Date: 2026-05-18 (round 3 after Phase 1a revision applying all 7 round-2 must-fix items)
Reviewed: ADAPTER_DESIGN_PHASE1A.md (66823 bytes, 918 lines)

## Verdicts

| Reviewer | Status | Verdict | Bytes |
|---|---|---|---|
| codex-ask | DONE | CONDITIONAL — 2 text fixes; "ready to implement after those two" | 637 |
| cursor-agent | DONE | CONDITIONAL — 2 items; "not ready to implement until reconciled" | 1014 |
| perplexity-ask | FAILED (banner only — broken for all 3 rounds) | Not usable | 83 |

2 of 3 review arms produced usable verdicts. Both converge on the same 2 issues.

## Convergent must-fix items

### Issue 1: §7 (Phase 1.5 isolation) still lists HIGH leak in-scope for Phase 1a
Codex: "§7 still lists 'HIGH leak event narration' under 'Allowed in Phase 1a CFO dashboard adapter,' directly contradicting §3, §9.1, §10, §12, and §14."
Cursor: "§7 'Phase 1.5 isolation' — Under 'Allowed in Phase 1a CFO dashboard adapter' it still lists HIGH leak event narration and the generic /api/events/speak proxy"
Diagnosis: Stale text in §7 bullet list — HIGH leak was removed from §3/§9.1/§10/§12/§14 but not from §7.
Fix: Replace HIGH leak references in §7 with "Sync-all final voice summary only". Remove the generic /api/events/speak proxy reference.

### Issue 2: §5.1 auth/startup rule is internally inconsistent
Cursor: "§5.1 startup/auth rule — Bullet 1 ties default TTS_API_KEY requirement to non-loopback bind only; bullet 3 and lines 690-691 read as uniform (loopback and non-loopback): refuse start if key unset and ALLOW_UNAUTH_TTS!=1. Pick one coherent rule and align all bullets."
Codex: §12 checklist contradicts §7 (auto-resolves once §7 is fixed).
Diagnosis: Auth spec written two ways — bullet 1 says non-loopback-only, bullets 3+ say uniform.
Fix: Pick one (recommend uniform: refuse start if key unset and ALLOW_UNAUTH_TTS!=1, regardless of bind interface, since dev/test escape hatch covers loopback dev work).

## Architecture status

All 7 round-2 must-fix items landed in their intended sections:
- R2-1 (TTS_API_KEY bypass) -> §5.1 (still has internal contradictions per Issue 2)
- R2-2 (/api/events/speak raw text) -> §3 (clean)
- R2-3 (HIGH leak removal) -> §3, §9.1, §10, §12, §14 clean, §7 STALE (Issue 1)
- R2-4 (adapter fail-closed) -> §2, §3 (clean)
- R2-5 (X-Caller server-side logging) -> §1 (clean)
- R2-6 (voice_text_utils tests) -> §4 (clean)
- R2-7 (CFO allowlist tight) -> §3 (clean, locked to ["sync_all_summary"])

No new architectural must-fix items. Both remaining issues are text-cleanup bugs.

## Health snapshot (post-revision)

- voice_tts: healthy on CUDA, voices=["mike"], audio_cache_size=4
- voicebox: healthy (legacy failover, pytorch/cpu)
- jax-bot: PID 1535697 unchanged (screen 'jax', canonical pathway)
- No source/code/service/Docker/systemd/Cloudflare/JAX/dashboard files touched during round 3

## Recommendation

Apply the 2 text fixes to ADAPTER_DESIGN_PHASE1A.md:
1. §7 — remove HIGH leak from "Allowed in Phase 1a" bullet list; replace with "Sync-all final voice summary only"; remove generic /api/events/speak proxy reference
2. §5.1 — unify auth rule: "voice_tts refuses to start if TTS_API_KEY unset and ALLOW_UNAUTH_TTS!=1, regardless of bind interface"

Then optionally re-run codex-ask only (quick sanity check, since these are text-only fixes). If verdict GO, implementation begins per §11 order.

This is the smallest revision yet — likely a single-file str_replace pass. Could be done with str_replace tool directly without invoking Claude Code.
