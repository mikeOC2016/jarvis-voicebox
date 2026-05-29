#!/bin/bash
set -eu
LOG=~/projects/voice_tts/logs/01_probe.log
STATUS=~/projects/voice_tts/logs/01_probe.status
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
finish() { echo "$1" > "$STATUS"; log "FINAL_STATUS: $1"; }
trap 'finish FAILED' ERR
echo RUNNING > "$STATUS"
log "=== PHASE 1: Docker + NGC + GPU probe ==="
log "Step 1.1: pull small NGC cuda image (test public-pull)"
docker pull nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 2>&1 | tee -a "$LOG"
log "Step 1.2: GPU visibility inside container"
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi 2>&1 | tee -a "$LOG"
log "Step 1.3: pull NGC pytorch:25.10-py3 (~10GB, long step)"
time docker pull nvcr.io/nvidia/pytorch:25.10-py3 2>&1 | tee -a "$LOG"
log "Step 1.4: torch + CUDA sanity inside pytorch:25.10-py3"
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.10-py3 python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'dev=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A','cap=',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | tee -a "$LOG"
finish OK
