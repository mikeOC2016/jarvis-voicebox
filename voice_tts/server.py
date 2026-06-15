"""
server.py - XTTS-v2 voice cloning TTS service for PGX.

Endpoints:
  GET  /health           - liveness + model status (unauthenticated)
  GET  /voices           - list available reference voices
  POST /speak            - synthesize full WAV (returns audio/wav)
  POST /speak/stream     - stream synthesized audio chunks

Auth: bearer token via TTS_API_KEY env (all endpoints except /health).

Optimizations:
  - XTTS model loaded once on startup, kept in GPU memory.
  - Speaker latents (gpt_cond_latent + speaker_embedding) precomputed per voice
    on first use, then cached. This is the single biggest latency win.
  - LRU cache of recently synthesized clips (key = hash of text+voice+lang).
  - Streaming endpoint uses model.inference_stream for sub-400ms first-byte.
"""
from __future__ import annotations

import io
import os
import sys
import time
import wave
import logging
import secrets
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import xxhash
from text_normalize import normalize_tts_text
import orjson

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Quiet down some noisy libs before TTS imports.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("voice_tts")

# ---------------------------------------------------------------------------
# torchaudio.load monkeypatch (Codex 2026-05-18 verdict C).
#
# torchaudio 2.9.1's load() routes through load_with_torchcodec which hard-
# requires the torchcodec native library. torchcodec 0.12.0's .so does not
# dlopen on aarch64 against torch 2.9.1+cu130 — every libtorchcodec_coreN.so
# fails with either a missing FFmpeg libavutil soname or
# `undefined symbol: torch_get_mutable_data_ptr` (torch ABI mismatch).
#
# For our use case — XTTS-v2 reading reference .wav clips during latent
# precompute — soundfile (libsndfile) is a drop-in backend. Patch is applied
# at module import time, BEFORE TTS is imported lazily inside TTSEngine.load.
# A loud guard fails fast if anything inverts that order.
# ---------------------------------------------------------------------------
if "TTS" in sys.modules:
    raise RuntimeError(
        "FATAL: torchaudio.load monkeypatch must apply before TTS is imported; "
        "TTS already in sys.modules. Re-check import ordering in server.py."
    )

_apex_original_load = torchaudio.load


def _apex_soundfile_load(
    filepath,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,  # noqa: ARG001  (soundfile float32 is already normalized)
    channels_first: bool = True,
    format=None,  # noqa: ARG001  (libsndfile autodetects)
    buffer_size: int = 4096,  # noqa: ARG001
):
    """soundfile-backed drop-in for torchaudio.load. Returns (Tensor, sr)."""
    start = int(frame_offset) if frame_offset else 0
    frames = int(num_frames) if num_frames is not None and num_frames > 0 else -1
    data, sr = sf.read(
        str(filepath),
        start=start,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    # soundfile returns (samples, channels); torchaudio convention is channels-first.
    arr = np.ascontiguousarray(data.T if channels_first else data)
    return torch.from_numpy(arr), int(sr)


torchaudio.load = _apex_soundfile_load
print(
    "[voice_tts startup] torchaudio.load patched -> soundfile backend "
    "(codec bypass per Codex verdict C 2026-05-18)",
    file=sys.stderr,
    flush=True,
)

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/voices"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))
MODEL_NAME = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "mike")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en")
API_KEY = os.environ.get("TTS_API_KEY", "")
ALLOW_UNAUTH_TTS = os.environ.get("ALLOW_UNAUTH_TTS", "")
LRU_CAPACITY = int(os.environ.get("LRU_CAPACITY", "1000"))
WARMUP_TEXT = os.environ.get("WARMUP_TEXT", "Service ready.")

# ---------------------------------------------------------------------------
# Prosody controls (audit P1 #5). XTTS on raw defaults sounds robotic; these
# give a calmer, more natural "JARVIS" cadence and are env-overridable. Per
# request overrides are also accepted (SpeakRequest). enable_text_splitting
# does sentence-level chunking so long input gets natural pauses instead of one
# rushed flat block.
# ---------------------------------------------------------------------------
PROSODY_DEFAULTS = {
    "temperature": float(os.environ.get("XTTS_TEMPERATURE", "0.70")),
    "length_penalty": float(os.environ.get("XTTS_LENGTH_PENALTY", "1.0")),
    "repetition_penalty": float(os.environ.get("XTTS_REPETITION_PENALTY", "2.5")),
    "top_k": int(os.environ.get("XTTS_TOP_K", "50")),
    "top_p": float(os.environ.get("XTTS_TOP_P", "0.85")),
    "speed": float(os.environ.get("XTTS_SPEED", "1.0")),
}
ENABLE_TEXT_SPLITTING = os.environ.get("XTTS_ENABLE_TEXT_SPLITTING", "1") == "1"


# ---------------------------------------------------------------------------
# §5.1 startup guard (ADAPTER_DESIGN_PHASE1A.md Rule 23 round 2 R2-1).
#
# voice_tts REQUIRES TTS_API_KEY by default on every bind interface. The
# only escape hatch is the explicit ALLOW_UNAUTH_TTS=1 env var (dev/test).
# No ENVIRONMENT-based conditional, no "production" keyword. The guard runs
# at module import time, BEFORE engine.load() / XTTS / GPU allocation.
# ---------------------------------------------------------------------------

def _enforce_startup_auth_policy() -> None:
    if API_KEY:
        return
    if ALLOW_UNAUTH_TTS == "1":
        log.warning(
            "[TTS-STARTUP] WARNING: running with unauthenticated /speak; "
            "ALLOW_UNAUTH_TTS=1 set explicitly"
        )
        return
    log.error(
        "[TTS-STARTUP] refusing to start: "
        "TTS_API_KEY is unset and ALLOW_UNAUTH_TTS!=1"
    )
    sys.exit(1)


_enforce_startup_auth_policy()

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _f32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    a = np.clip(audio, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_f32_to_pcm16_bytes(audio))
    return buf.getvalue()


def _wav_header(sr: int, data_bytes: int = 0xFFFFFFFF - 36) -> bytes:
    """WAV header for streaming - use a giant data size since we don't know length up front."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00" * 0)
    header = buf.getvalue()
    # Patch sizes in-place
    header = bytearray(header)
    # RIFF chunk size (offset 4): total file size - 8
    riff_size = (36 + data_bytes).to_bytes(4, "little")
    header[4:8] = riff_size
    # data chunk size (offset 40): data_bytes
    header[40:44] = data_bytes.to_bytes(4, "little")
    return bytes(header)

# ---------------------------------------------------------------------------
# Model wrapper with speaker-latent + LRU caching
# ---------------------------------------------------------------------------

class _LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get_(self, k):
        if k not in self:
            return None
        self.move_to_end(k)
        return self[k]

    def put_(self, k, v):
        if k in self:
            self.move_to_end(k)
        self[k] = v
        while len(self) > self.capacity:
            self.popitem(last=False)


class TTSEngine:
    def __init__(self):
        self.model = None
        self.sample_rate = 24000  # XTTS-v2 native output sample rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_latents: dict[str, tuple] = {}
        self.audio_cache = _LRU(LRU_CAPACITY)
        self._lock = threading.Lock()
        # Single GPU inference gate: the XTTS model is shared across FastAPI
        # worker threads and is NOT safe to run concurrently. Serialize every
        # inference / inference_stream call through this lock to avoid CUDA OOM,
        # hangs, and corrupted audio under dashboard bursts (audit P1 #5 P0).
        self._infer_lock = threading.Lock()
        self._ready = False
        self._loaded_at = None

    @staticmethod
    def _resolve_prosody(
        speed: float | None,
        temperature: float | None,
        top_p: float | None,
        repetition_penalty: float | None,
    ) -> dict:
        p = dict(PROSODY_DEFAULTS)
        if speed is not None:
            p["speed"] = speed
        if temperature is not None:
            p["temperature"] = temperature
        if top_p is not None:
            p["top_p"] = top_p
        if repetition_penalty is not None:
            p["repetition_penalty"] = repetition_penalty
        return p

    def load(self):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        log.info("Resolving %s via ModelManager...", MODEL_NAME)
        mgr = ModelManager()
        model_path, config_path, _ = mgr.download_model(MODEL_NAME)

        config = XttsConfig()
        config.load_json(str(Path(model_path) / "config.json"))
        log.info("Loading XTTS-v2 from %s (device=%s)", model_path, self.device)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_path), eval=True)
        if self.device == "cuda":
            model.cuda()
        self.model = model
        self.sample_rate = int(getattr(config.audio, "output_sample_rate", 24000))
        log.info("XTTS-v2 loaded. sample_rate=%d", self.sample_rate)

        for voice_name in self.list_voices():
            try:
                self._compute_latents(voice_name)
            except Exception:
                log.exception("failed to precompute latents for voice=%s", voice_name)

        log.info("Warmup synthesis: %r", WARMUP_TEXT)
        try:
            self.synthesize(WARMUP_TEXT, voice=DEFAULT_VOICE, language=DEFAULT_LANG)
            log.info("Warmup OK")
        except Exception:
            log.exception("warmup failed (non-fatal)")

        self._ready = True
        self._loaded_at = time.time()

    def list_voices(self) -> list[str]:
        if not VOICES_DIR.exists():
            return []
        return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))

    def voice_path(self, voice: str) -> Path:
        # Confine to VOICES_DIR: a voice is a single bare filename stem, never a
        # path. Reject separators, traversal, NUL, and anything that resolves
        # outside the voices directory (path traversal hardening).
        if not voice or voice in (".", "..") or "\x00" in voice:
            raise HTTPException(400, "invalid voice name")
        if "/" in voice or "\\" in voice or os.sep in voice or voice != Path(voice).name:
            raise HTTPException(400, "invalid voice name")
        base = VOICES_DIR.resolve()
        p = (base / f"{voice}.wav").resolve()
        if p.parent != base:
            raise HTTPException(400, "invalid voice name")
        if not p.exists():
            raise HTTPException(404, f"voice not found: {voice}")
        return p

    def _compute_latents(self, voice: str):
        path = self.voice_path(voice)
        log.info("Computing speaker latents for voice=%s from %s", voice, path)
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=str(path)
        )
        self.speaker_latents[voice] = (gpt_cond_latent, speaker_embedding)

    def _latents_for(self, voice: str):
        if voice not in self.speaker_latents:
            self._compute_latents(voice)
        return self.speaker_latents[voice]

    @staticmethod
    def _cache_key(text: str, voice: str, language: str) -> str:
        h = xxhash.xxh3_64()
        h.update(text.encode("utf-8"))
        h.update(b"\0")
        h.update(voice.encode("utf-8"))
        h.update(b"\0")
        h.update(language.encode("utf-8"))
        return h.hexdigest()

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str,
        *,
        speed: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[np.ndarray, int]:
        if not self.model:
            raise HTTPException(503, "model not loaded")
        normalized_text = normalize_tts_text(text)
        key = self._cache_key(normalized_text, voice, language)
        with self._lock:
            cached = self.audio_cache.get_(key)
        if cached is not None:
            return cached, self.sample_rate

        gpt_cond_latent, speaker_embedding = self._latents_for(voice)
        prosody = self._resolve_prosody(speed, temperature, top_p, repetition_penalty)
        # Serialize GPU inference: shared XTTS model is not concurrency-safe.
        with self._infer_lock:
            out = self.model.inference(
                text=normalized_text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=ENABLE_TEXT_SPLITTING,
                **prosody,
            )
        audio = np.asarray(out["wav"], dtype=np.float32)

        with self._lock:
            self.audio_cache.put_(key, audio)
        return audio, self.sample_rate

    def synthesize_stream(
        self,
        text: str,
        voice: str,
        language: str,
        *,
        speed: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[bytes]:
        if not self.model:
            raise HTTPException(503, "model not loaded")
        normalized_text = normalize_tts_text(text)
        gpt_cond_latent, speaker_embedding = self._latents_for(voice)
        prosody = self._resolve_prosody(speed, temperature, top_p, repetition_penalty)
        # Hold the inference gate for the lifetime of the stream so a single GPU
        # serves one stream at a time (audit P1 #5 P0).
        with self._infer_lock:
            chunks = self.model.inference_stream(
                text=normalized_text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=ENABLE_TEXT_SPLITTING,
                **prosody,
            )
            yield _wav_header(self.sample_rate)
            for chunk in chunks:
                if torch.is_tensor(chunk):
                    arr = chunk.detach().cpu().numpy().astype(np.float32)
                else:
                    arr = np.asarray(chunk, dtype=np.float32)
                yield _f32_to_pcm16_bytes(arr)


engine = TTSEngine()

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="voice_tts", version="1.0.0")


# §5.2 X-Caller allowlist (ADAPTER_DESIGN_PHASE1A.md Rule 23 round 2 R2-5).
# Hard-locked, case-sensitive. Round-1 names "dashboard-events" and
# "hub-events" are retired and intentionally absent.
_ALLOWED_CALLERS = frozenset({
    "jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub",
})


def _sanitize_caller(value: Optional[str]) -> str:
    if value is None or value not in _ALLOWED_CALLERS:
        return "unknown"
    return value


@app.middleware("http")
async def _log_tts_calls(request: Request, call_next):
    """Emit exactly one [TTS] log line per /speak{,/stream} call.

    Format is byte-identical to the §5.2 spec. Spoken text body and bearer
    token are never read here and never logged.
    """
    if request.url.path not in ("/speak", "/speak/stream"):
        return await call_next(request)
    t0 = time.time()
    caller = _sanitize_caller(request.headers.get("X-Caller"))
    response = await call_next(request)
    latency_ms = int((time.time() - t0) * 1000)
    log.info(
        "[TTS] caller=%s status=%d latency_ms=%d",
        caller, response.status_code, latency_ms,
    )
    return response


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str = Field(DEFAULT_VOICE)
    language: str = Field(DEFAULT_LANG)
    # Optional prosody overrides; None falls back to PROSODY_DEFAULTS.
    speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    temperature: Optional[float] = Field(None, ge=0.1, le=1.5)
    top_p: Optional[float] = Field(None, ge=0.1, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=10.0)


def require_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    presented = authorization.removeprefix("Bearer ").strip()
    # Constant-time comparison: avoids leaking the key via timing side channel.
    if not secrets.compare_digest(presented, API_KEY):
        raise HTTPException(403, "invalid token")


@app.on_event("startup")
def _startup():
    try:
        engine.load()
    except Exception:
        log.exception("FATAL: engine.load() failed")
        # Let the process come up unhealthy so docker healthcheck flags it
        engine._ready = False


@app.get("/health")
def health():
    return {
        "status": "ok" if engine._ready else "loading",
        "model": MODEL_NAME,
        "model_loaded": engine._ready,
        "gpu_available": torch.cuda.is_available(),
        "device": engine.device,
        "sample_rate": engine.sample_rate,
        "voices": engine.list_voices(),
        "speaker_latents_cached": list(engine.speaker_latents.keys()),
        "audio_cache_size": len(engine.audio_cache),
        "loaded_at": engine._loaded_at,
    }


@app.get("/voices", dependencies=[Depends(require_api_key)])
def voices():
    return {"voices": engine.list_voices()}


@app.post("/speak", dependencies=[Depends(require_api_key)])
def speak(req: SpeakRequest):
    t0 = time.time()
    audio, sr = engine.synthesize(
        req.text,
        req.voice,
        req.language,
        speed=req.speed,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    wav = _wav_bytes(audio, sr)
    elapsed_ms = int((time.time() - t0) * 1000)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Synthesis-Ms": str(elapsed_ms),
            "X-Voice": req.voice,
            "X-Language": req.language,
            "X-Sample-Rate": str(sr),
            "X-Audio-Bytes": str(len(wav)),
            "Cache-Control": "no-store",
        },
    )


@app.post("/speak/stream", dependencies=[Depends(require_api_key)])
def speak_stream(req: SpeakRequest):
    gen = engine.synthesize_stream(
        req.text,
        req.voice,
        req.language,
        speed=req.speed,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    return StreamingResponse(
        gen,
        media_type="audio/wav",
        headers={
            "X-Voice": req.voice,
            "X-Language": req.language,
            "X-Sample-Rate": str(engine.sample_rate),
            "Cache-Control": "no-store",
            # Disable Cloudflare buffering
            "X-Accel-Buffering": "no",
            "CF-Cache-Status": "BYPASS",
        },
    )
