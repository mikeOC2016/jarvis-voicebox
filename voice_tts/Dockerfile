# voice_tts - Strategy B per Codex 2nd verdict
# Public PyTorch cu130 aarch64 trio, ABI-consistent, on NVIDIA CUDA 13 runtime.
FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    TTS_HOME=/cache/tts \
    COQUI_TOS_AGREED=1 \
    PYTHONPATH=/opt/voice_tts

# System deps: Python 3.12 + venv + audio libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libsndfile1 ffmpeg curl ca-certificates \
        build-essential && \
    python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/voice_tts

# PyTorch cu130 aarch64 trio - all built together so ABI is consistent.
RUN /opt/venv/bin/pip install \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu130

# App + coqui-tts requirements
COPY requirements.txt /opt/voice_tts/requirements.txt
RUN /opt/venv/bin/pip install -r /opt/voice_tts/requirements.txt && \
    /opt/venv/bin/pip install coqui-tts==0.27.5

# Bypass coqui-tts defensive torchcodec import check (Codex 2026-05-18 verdict B).
# Probe finding: torchcodec 0.12.0 has a real aarch64 wheel but its native
# libtorchcodec_core4.so has an ABI mismatch with torch 2.9.1 cu130 — .so will
# not dlopen. XTTS-v2 inference uses the torchaudio spectrogram path and never
# calls codec ops at runtime, so we patch out the defensive check rather than
# ship a known-broken native lib. GUARDED: build fails if the target line shape
# changes (Coqui upgrade) so we know to reassess instead of silently shipping
# with a broken patch.
RUN TTS_INIT=/opt/venv/lib/python3.12/site-packages/TTS/__init__.py && \
    grep -qF '        raise ImportError(TORCHCODEC_IMPORT_ERROR)' "$TTS_INIT" || \
        (echo "FATAL: TTS/__init__.py changed shape; codec patch needs reassessment" >&2 && exit 1) && \
    sed -i 's|        raise ImportError(TORCHCODEC_IMPORT_ERROR)|        pass  # XTTS spec-path only; codec bypass per Codex 2026-05-18|' "$TTS_INIT" && \
    grep -qF 'XTTS spec-path only; codec bypass' "$TTS_INIT" && \
    echo "TTS codec defensive check patched OK"

# Validate at build time. torchaudio op runs on CPU (no GPU at build time) but proves ABI is sound.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda_built', torch.version.cuda)" && \
    python -c "import torchaudio; print('torchaudio', torchaudio.__version__); import torch; t=torch.zeros(16000); spec=torchaudio.transforms.MelSpectrogram(sample_rate=16000)(t); print('torchaudio op OK shape', spec.shape)" && \
    python -c "from TTS.api import TTS; print('TTS api OK')" && \
    python -c "import fastapi, uvicorn, soundfile, librosa, noisereduce, pyloudnorm; print('audio + server deps OK')"

COPY server.py prep_reference.py text_normalize.py /opt/voice_tts/

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD curl -fsS http://localhost:8050/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8050", "--workers", "1"]
