#!/bin/bash
# Phase 3: smoke test - synthesize three phrases, measure latency.
set -eu
cd "$(dirname "$0")/.."
KEY=$(grep ^TTS_API_KEY= .env | cut -d= -f2-)
OUTDIR=test_outputs
mkdir -p "$OUTDIR"
LOG=logs/03_smoke.log
STATUS=logs/03_smoke.status
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
echo RUNNING > "$STATUS"
log "=== PHASE 3: smoke test ==="

run_speak() {
  local label="$1"; shift
  local text="$*"
  local out="$OUTDIR/$label.wav"
  log "-> $label : $text"
  local t0=$(date +%s.%N)
  local sz
  http_code=$(curl -sS -o "$out" -w '%{http_code}' \
      -H "Authorization: Bearer $KEY" \
      -H 'Content-Type: application/json' \
      -D "$OUTDIR/$label.headers" \
      --data "{\"text\":\"$text\",\"voice\":\"mike\",\"language\":\"en\"}" \
      http://127.0.0.1:8050/speak)
  local t1=$(date +%s.%N)
  local dur=$(python3 -c "print(round($t1-$t0,3))")
  sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
  log "   http=$http_code elapsed=${dur}s bytes=$sz"
  grep -i '^x-synthesis-ms' "$OUTDIR/$label.headers" | tee -a "$LOG" || true
}

run_speak apex_killswitch "APEX kill-switch activated. All positions flat."
run_speak cfo_online "CFO dashboard online. Plaid sync complete."
run_speak jax_ready "JAX bot ready. Standing by."

log "=== smoke test outputs ==="
ls -la "$OUTDIR" | tee -a "$LOG"
echo OK > "$STATUS"
log FINAL_STATUS: OK
