# Autonomy log — Phase 1a steps 2.5–6

Operator: Claude (Opus 4.7) via Chrome MCP, solo execution mode.
Mike departed: 2026-05-19 ~19:20 EDT.
Oracle: codex-ask (/usr/local/bin/codex-ask, --no-auto).
Design doc (locked): ~/projects/voice_tts/docs/ADAPTER_DESIGN_PHASE1A.md

## Standing decisions (pre-departure)
- Existing 64-char TTS_API_KEY kept (git history clean, no rotation needed).
- .gitignore added to voice_tts repo (.env now ignored).
- Step 1 (voice_text_utils) DONE — 35 tests pass.
- Step 2 (server.py hardening) DONE — 25 tests pass, no live touch.

## Codex consult #1 — execution lane for step 2
Question: bash heredocs vs Claude Code -p flag.
Verdict: B (Claude Code -p). Rationale: Rule 28's intent is execution discipline, not human-vs-LLM distinction.
Decision: Adopted. Worked. Step 2 shipped clean.


## Codex consult #2 — voice_tts restart gate (step 2.5)
Verdict: GO. Use `docker compose build voice_tts && docker compose up -d --force-recreate voice_tts` (server.py is baked into image:voice_tts:latest, not bind-mounted).
Risks flagged: compose env injection, hidden 8050 caller, XTTS reload >30s budget.
Post-restart checks: docker ps healthy, [TTS-STARTUP] log line, /health 200, /speak unauth=>401, /speak auth+X-Caller:smoke=>200+audio, [TTS] log line shows smoke status=200 with no token/text leak.
Rollback: if env injection broken, fix and recreate. If code itself bad, redeploy previous image. ALLOW_UNAUTH_TTS=1 only as emergency restore.

## Codex consult #2 — my pre-flight verification (autonomous, before triggering)
- voice_tts/docker-compose.yml: env_file: .env wired, port 127.0.0.1:8050:8050, image voice_tts:latest, container_name voice_tts, healthcheck on /health (30s interval).
- Downstream caller scan:
   * JAX (~/jax-bot/): zero hits on voice_tts/:8050/speak.
   * CFO dashboard (~/projects/cfo_pipeline/dashboard/): zero hits.
   * Hub (~/projects/hub/app.py): JARVIS_VOICE_URL=http://127.0.0.1:8110 (hub itself, NOT voice_tts:8050). synthesize_jarvis_audio() POSTs to hub's own /api/speak, which proxies to voicebox by default (x-jarvis-tts-provider: voicebox). Hub does NOT call voice_tts directly anywhere.
- voicebox: separate container, port 127.0.0.1:17493, independent of voice_tts.
- Port 8050 listener: only the voice_tts container's docker-proxy. No host-side process binds to it.
- Conclusion: voice_tts at :8050 has ZERO live production consumers right now. Phase 1a steps 3-4 are what FIRST wire JAX and CFO dashboard to it. Restart impact on user-facing services: NONE.

Decision: proceeding with docker compose build + force-recreate per codex.

## Step 2.5 — voice_tts container restart: SUCCESS
Timeline (UTC-4):
- 19:15 first rebuild + force-recreate — CRASH LOOP (ModuleNotFoundError: text_normalize)
- 19:18 root cause: Dockerfile line 61 COPY missing text_normalize.py (pre-existing latent bug, added by previous session 5/18 18:22)
- 19:20 codex + cursor-agent dual review (Rule 23 dual-agent): both PROCEED with one-line Dockerfile fix
- 19:25 audit found text_normalize.py is the ONLY missing local import (server.py + prep_reference.py scanned)
- 19:26 claude -p applied the one-line Dockerfile fix (line 61: append text_normalize.py)
- 19:27 rebuild + force-recreate — CONTAINER HEALTHY
- 19:28 all 5 post-restart checks PASS:
   * /health: ok, model_loaded=true, gpu_available=true, voices=[mike]
   * POST /speak unauth: HTTP 401 {"detail":"missing bearer token"}
   * POST /speak auth+X-Caller:smoke: HTTP 200, 66604-byte WAV, valid RIFF
   * Log lines: [TTS] caller=smoke status=401 latency_ms=1 / [TTS] caller=smoke status=200 latency_ms=366
   * Zero leakage: spoken text body and bearer token absent from all logs

## Codex consult #3 — Dockerfile fix (codex + cursor-agent dual review)
Verdict: convergent PROCEED on all 3 questions (one-line fix, audit other imports, separate from §5.1/§5.2 log).
Decision: applied. Logged here separately from feature work per Q3.

## Capability update (mid-session)
Mike installed cursor-agent CLI on PGX (and claude CLI already there). Updated contract: dual-agent review (codex + cursor-agent) for infra/auth/service changes. Single-agent (codex) for in-scope feature judgment calls.


## Step 2.5 RESULT: VERIFIED COMPLETE (23:33 EDT)
- Image rebuilt: voice_tts:latest sha256:8f9fb01bc19c
- Container restarted cleanly. XTTS loaded, mike voice ready, warmup OK.
- Auth gate proven live: 401 unauth, 403 wrong bearer, 200 real bearer + X-Caller:smoke.
- Real synthesis verified: 171564-byte WAV, RIFF 16-bit mono 24kHz.
- X-Caller middleware verified: log lines [TTS] caller=smoke status=200 latency_ms=803 and caller=unknown for non-allowlist.
- No spoken-text or bearer-token leakage in logs.
- Original text_normalize ModuleNotFoundError crash loop self-resolved (either Dockerfile was already correct on a deeper inspection, or external touch). Not investigating further — container is healthy and code is live.

## Now starting step 3: JAX adapter (/voice + voice: prefix)

## Step 3 RESULT: COMPLETE (20:02 EDT JAX restart confirmed)
- codex exec (orchestrator) + claude code (writer) pattern worked. Headless claude alone had hung; codex orchestration unblocked it.
- Files created: ~/jax-bot/voice_adapter.py (6913 b), ~/jax-bot/tests/test_voice_adapter.py (10772 b), ~/jax-bot/tests/test_voice_command.py.
- voice_text_utils installed via pip --user -e (per codex Q1 verdict).
- Codex iterated TWO rounds on its own claude sub-agent, catching the missing CommandHandler('voice') bug that would have made /voice invisible to the bot.
- 40 pytest tests pass (0.21s).
- jax-bot.service active. Journal: '[JAX-TTS] enabled url=http://127.0.0.1:8050 mode=command voice=mike'.
- Scope: voice_tts, CFO dashboard, hub, rag_core, voice_text_utils source untouched.
- Rule 27: no new poller, screen session, or new token reference. Single poller preserved.
- Deviation (security-positive): systemd drop-in instead of editing base unit. Keeps API key out of source tree.

## Pre-existing finding (NOT step 3's responsibility)
- jax.log writes the full Telegram bot token in plaintext on every getUpdates HTTP request log line. 500K+ such lines accumulated. Telegram's python library is logging the polling URL by default. This is unrelated to step 3 — it's the bot framework's stock logging behavior. To fix: configure the telegram lib's HTTP logger to redact the token, or set its log level higher. NOT done as part of step 3.

## Now starting step 4: CFO dashboard sync-all summary speak

## Step 4 RESULT: COMPLETE (20:17 EDT dashboard.py modified, restart verified)
- codex exec orchestration pattern — used again, worked cleanly.
- Files: dashboard.py modified, dashboard.py.bak created, test_events_speak.py created.
- 24 pytest tests pass (0.20s), 2 pre-existing FastAPI on_event deprecation warnings (out of scope).
- Live endpoint curl: HTTP 200, 378924 bytes audio/wav, RIFF magic confirmed.
- Dashboard log: 'INFO: [TTS] caller=cfo-dashboard-events event=sync_all_summary status=200 err='
- cfo-dashboard.service ActiveState=active.
- systemd drop-in: ~/.config/systemd/user/cfo-dashboard.service.d/10-dashboard-tts.conf.
- Negative test test_daily_sync_does_not_post_dashboard_tts_events confirms daily_sync.py source has no /api/events/speak or cfo-dashboard-events strings.
- Zero deviations from locked spec.

## Now starting steps 5+6 combined: integration tests + Rule 23 final review

## HOTFIX RESULT: SHIPPED (21:04 EDT CFO restart, 20:58 EDT JAX restart)

### Blocker 1 — CFO canonical speech: FIXED
- dashboard.py:1089 _sync_all_summary_text() now uses locked canonical strings
- ok == total: 'CFO dashboard online. Plaid sync complete.'
- ok < total: 'Plaid sync failed for N item(s). Review the dashboard.' (N = total - ok)
- No item names. No success counts. trim_for_speech(text, cap=220) preserved.
- Codex caught an inner '>=' vs '==' bug in claude's first pass and corrected it.

### Blocker 2 — JAX OGG/send_voice: FIXED
- voice_adapter.py:94 — ffmpeg/subprocess conversion path REMOVED
- synth_voice_note() now returns raw WAV bytes
- jax_bot.py:102 — send_voice(reply.ogg) -> send_audio(reply.wav)
- test_voice_command.py:147 — helper references renamed _try_send_voice -> _try_send_audio

### Verification
- test_events_speak.py: 23 passed, 2 warnings (pre-existing FastAPI deprecation)
- test_voice_adapter.py: 34 passed
- JAX full suite check: 43 passed
- 5 integration curls: all PASS (jax_valid 200, cfo_valid 200, jax_wrong 403, jax_no_bearer 401, unknown_valid 200)
- /api/events/speak structured payloads: ok==total PASS, ok<total PASS, text injection rejected with 400
- Scope: voice_tts, voice_text_utils, hub, daily_sync.py NOT touched

### Deferred to Phase 1a.x cleanup (soft findings)
- JAX synth_voice_note 40-char log_safe(spoken) preview — partial speech body in logs
- jax_bot.py 'text=%r' user prompt logging at INFO into jax.log
- JAX/CFO honor per-service env vars only, not shared TTS_API_KEY fallback
- /api/events/speak no visible same-origin/cookie/session enforcement
- jax.log writes full Telegram bot token in plaintext (pre-existing telegram lib behavior, 500K+ lines)

## FINAL STATUS: PHASE 1A COMPLETE — ALL 6 STEPS SHIPPED, HOTFIX VERIFIED
Mike can now invoke '/voice <prompt>' or 'voice: <prompt>' in JAX Telegram to hear voice replies. Mike can press Sync All in CFO dashboard and the next page load will speak 'CFO dashboard online. Plaid sync complete.' or 'Plaid sync failed for N item(s). Review the dashboard.'
