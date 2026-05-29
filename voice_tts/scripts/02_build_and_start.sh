#!/bin/bash
# Phase 2: build image, start service, wait for healthy. Idempotent.
set -eu
cd "$(dirname "$0")/.."
LOG=logs/02_build.log
STATUS=logs/02_build.status
mkdir -p logs voices cache
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
finish() { echo "$1" > "$STATUS"; log "FINAL_STATUS: $1"; }
trap 'finish FAILED' ERR
echo RUNNING > "$STATUS"
log "=== PHASE 2: build + start ==="
log "Step 2.1: docker compose build (uses BuildKit by default)"
time DOCKER_BUILDKIT=1 docker compose build 2>&1 | tee -a "$LOG"
log "Step 2.2: docker compose up -d"
docker compose up -d 2>&1 | tee -a "$LOG"
log "Step 2.3: waiting for /health (up to 300s)"
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8050/health > /tmp/health.json 2>/dev/null; then
    if grep -q '"status":"ok"' /tmp/health.json; then
      log "healthy on attempt $i"
      cat /tmp/health.json | tee -a "$LOG"; echo | tee -a "$LOG"
      finish OK; exit 0
    fi
  fi
  sleep 5
done
log "timeout waiting for /health"
docker compose logs --tail=200 voice_tts 2>&1 | tee -a "$LOG"
finish TIMEOUT
exit 1
