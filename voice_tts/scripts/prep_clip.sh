#!/bin/bash
# Run the audio prep pipeline INSIDE the built voice_tts container.
# Reads:  /voices/raw/mike_raw.wav (volume-mounted from ./voices/raw/)
# Writes: /voices/mike.wav         (volume-mounted to ./voices/)
set -euo pipefail
IN="${1:-/voices/raw/mike_raw.wav}"
OUT="${2:-/voices/mike.wav}"
DBG="${3:-/voices/raw/_prep}"
cd "$(dirname "$0")/.."
docker compose run --rm prep_clip "$IN" "$OUT" "$DBG"
