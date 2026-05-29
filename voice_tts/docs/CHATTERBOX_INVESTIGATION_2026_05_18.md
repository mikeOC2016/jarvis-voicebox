# Chatterbox Investigation ? 2026-05-18

Author: Codex
Gate status: investigation-only; no service wiring; no Docker/systemd/Cloudflare changes.

## 1. Why this exists

During the Voice TTS adapter design cycle, a separate package from `Z:\voice claude files\voicebox_human.tar.gz` was staged at `/home/msr8109/projects/voicebox_human`. It is not part of the canonical `voice_tts` XTTS-v2 service and was not an approved replacement.

This memo captures why the Phase 1 adapters remain targeted at `voice_tts` on `127.0.0.1:8050`, not Chatterbox / `voicebox_human`.

## 2. Existing canonical services

- `voice_tts` ? canonical PGX-local TTS service, XTTS-v2, CUDA, bound to `127.0.0.1:8050`, canonical voice `mike`.
- `voicebox` ? legacy service on `127.0.0.1:17493`, kept as Phase 1 failover only.
- `voicebox_human` ? staged package only, not a running service, no systemd unit, no Docker service, no adapter target.

## 3. Probe A ? package deploy result

Command path: `/home/msr8109/projects/voicebox_human/deploy_pgx.sh`

Observed result from `/tmp/voicebox_human_deploy.log`:

```text
loading Chatterbox model on cpu
model loaded in 33.3s (sr=24000)
generated 8.60s audio in 51.18s
wrote /tmp/voicebox_real.wav (8.60s @ 24000Hz)
21 passed, 1 warning in 3.68s
```

Fresh venv inspection:

```text
torch 2.6.0+cpu
cuda_available False
cuda_version None
device_count 0
```

Conclusion: the 51.18s generation time was caused by the package installing CPU torch, not by a proven Chatterbox limitation on PGX hardware.

## 4. Root cause of CPU install

`chatterbox-tts==0.1.7` declares hard pins on Python 3.12:

```text
torch==2.6.0; python_version < "3.14"
torchaudio==2.6.0; python_version < "3.14"
transformers==5.2.0
diffusers==0.29.0
```

The default `pip install torch torchaudio` path resolved to `torch 2.6.0+cpu` / `torchaudio 2.6.0` in the package venv, so `ChatterboxEngine(device="auto")` correctly resolved to CPU.

## 5. Probe B ? isolated CUDA viability probe

Probe shape: throwaway `nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04` container, `--gpus all`, Python 3.12 venv, install `chatterbox-tts`, then force-reinstall the same public cu130 torch lane used by `voice_tts`.

Status file: `/tmp/chatterbox_cuda_probe.status`
Log file: `/tmp/chatterbox_cuda_probe.log`

Key output:

```text
after chatterbox install torch 2.6.0+cpu cuda None available False
after chatterbox install torchaudio 2.6.0
--- force CUDA torch/torchaudio cu130 lane ---
chatterbox-tts 0.1.7 requires numpy<2.0.0,>=1.24.0; python_version < "3.13", but you have numpy 2.4.4 which is incompatible.
chatterbox-tts 0.1.7 requires torch==2.6.0; python_version < "3.14", but you have torch 2.9.1+cu130 which is incompatible.
chatterbox-tts 0.1.7 requires torchaudio==2.6.0; python_version < "3.14", but you have torchaudio 2.9.1 which is incompatible.
torch 2.9.1+cu130 cuda 13.0 available True count 1
device0 NVIDIA GB10
torchaudio 2.9.1
import chatterbox OK
LOAD_OK sr 24000 elapsed 6.66
GENERATE_OK shape (1, 28800) device cpu elapsed 5.17
```

Conclusion: Chatterbox can import, load, and generate with the cu130 torch lane on PGX, but only after overriding `chatterbox-tts` package pins. That is a dependency-override posture, not a production posture.

## 6. Risk assessment

Chatterbox is viable enough to investigate later, but not viable enough to replace `voice_tts` now.

Specific risks:

- Hard package pins conflict with the PGX CUDA 13 / torch 2.9.1 lane.
- Forced dependency override mirrors the same class of ABI risk that appeared in the `torchcodec` / Coqui work.
- Probe generated successfully, but does not prove long-run stability, voice quality, streaming behavior, memory profile, or adapter API compatibility.
- The current `voicebox_human` package is not a service and has no production auth, health, metrics, cache, or adapter contract.

## 7. Decision

`voice_tts` remains the canonical Phase 1 / Phase 1a / Phase 1b / Phase 1.5 target.

`voicebox_human` remains a staged experimental package only. It is not a replacement for `voice_tts`, not a failover, and not referenced by adapter design docs.

`voicebox` remains the only legacy failover named in Phase 1 design.

## 8. Future gate

Any proposal to replace or augment `voice_tts` with Chatterbox requires a dedicated Rule 23 review titled:

```text
Chatterbox replace/augment voice_tts
```

Minimum review inputs:

1. A clean CUDA install path with no unowned torch/torchaudio/numpy conflicts, or a documented and reviewed override strategy.
2. A production service wrapper comparable to `voice_tts` (`/health`, auth, bounded request schema, cache, logging, caller attribution).
3. Real Mike reference-voice quality comparison against XTTS-v2.
4. Cold and warm latency table on PGX for short phrases and 300-character JAX-style replies.
5. Memory and GPU utilization under repeated calls.
6. Clear migration plan that does not leave three simultaneous TTS services in adapter paths.

Until that review produces GO, do not wire `voicebox_human` to JAX, CFO dashboard, MASTER DASHBOARD, Cloudflare, systemd, or any adapter.
