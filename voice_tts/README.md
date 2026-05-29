# voice_tts

Self-hosted XTTS-v2 voice cloning TTS service on PGX (DGX Spark GB10).

Serves CFO dashboard, APEX kill-switch alerts, and JAX bot via `tts.luxqozp.com`.

## Layout

```
voice_tts/
  Dockerfile              # NGC pytorch:25.10 base + coqui-tts 0.27.5
  docker-compose.yml      # --gpus all, volume mounts, healthcheck
  requirements.txt        # FastAPI + audio prep deps (NOT torch - from NGC)
  server.py               # FastAPI: /speak /speak/stream /health /voices
  prep_reference.py       # One-shot audio prep pipeline for reference clips
  voice-tts.service       # systemd unit (NOT installed - manual step)
  .env                    # TTS_API_KEY, defaults (mode 600)
  .env.example
  .dockerignore
  voices/
    mike_ref.wav          # Cleaned 15s reference clip (generated)
    raw/
      mike_raw.wav        # Raw upload (untouched)
      _prep/              # Per-stage debug audio (generated)
  cache/                  # HF cache + XTTS model + LRU audio cache
  scripts/
    01_docker_probe.sh    # PHASE 1: verify Docker + NGC + GPU
    02_build_and_start.sh # PHASE 2: build image, up -d, wait healthy
    03_smoke.sh           # PHASE 3: latency smoke test
    prep_clip.sh          # Run audio prep inside built container
  test_outputs/           # Smoke test .wav outputs
  logs/                   # Per-phase status + logs
```

## Build sequence

```bash
cd ~/projects/voice_tts
# Phase 1: probe (one-time)
bash scripts/01_docker_probe.sh
# Drop raw .wav at voices/raw/mike_raw.wav, OR use scp from MIKES-DESKTOOP
# Phase 2: build + start
bash scripts/02_build_and_start.sh
# Prep reference clip (one-time after build, before first real use)
bash scripts/prep_clip.sh
# Restart so model picks up the new mike_ref.wav
docker compose restart
# Phase 3: smoke test
bash scripts/03_smoke.sh
```

## API

All endpoints except `/health` require `Authorization: Bearer $TTS_API_KEY`.

```bash
# Health (no auth)
curl http://127.0.0.1:8050/health

# Synthesize (full WAV)
curl -X POST http://127.0.0.1:8050/speak \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello world","voice":"mike","language":"en"}' \
  --output hello.wav

# Stream (sub-400ms first byte)
curl -X POST http://127.0.0.1:8050/speak/stream ...
```

## Triple-review gate (Rule 23)

Before enabling Cloudflare route `tts.luxqozp.com`:
1. Cursor: review server.py + Dockerfile on PGX
2. Perplexity: architecture check
3. Codex: logic audit (initial Docker-vs-venv question already done: GO Strategy A)

## Codex risks (initial audit)

1. <400ms first-byte requires preload + cached latents + streaming. Implemented.
2. PyTorch 2.9+ needs torchcodec - Dockerfile validates the import.
3. NGC base /etc/pip/constraint.txt - install commands use --constraint to honor it.
4. Cloudflare tunnel can defeat streaming - server sets X-Accel-Buffering: no. Tunnel config TBD.
5. coqui-tts pinned to 0.27.5.
