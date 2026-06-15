"""
Tests for audit P1 #5: XTTS running on raw defaults + unsafe concurrency.

Audit (server.py TTSEngine.synthesize/synthesize_stream): the shared XTTS model
was called with no prosody controls and with no inference gate, so concurrent
dashboard bursts could corrupt audio / OOM the GPU, and output was robotic.

Fixes proven here:
  - synthesize passes prosody controls (temperature/top_p/repetition_penalty/
    speed/length_penalty/top_k) and enable_text_splitting to model.inference.
  - per-request overrides flow through.
  - a single inference lock serializes concurrent synthesize calls.

No GPU: model.inference is replaced by a fake that records kwargs/concurrency.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

TEST_BEARER_TOKEN = "prosody-test-token-DO-NOT-LOG"


@pytest.fixture(scope="module")
def server_module():
    os.environ["TTS_API_KEY"] = TEST_BEARER_TOKEN
    os.environ.pop("ALLOW_UNAUTH_TTS", None)
    import server  # noqa: WPS433

    server.API_KEY = TEST_BEARER_TOKEN
    return server


class FakeModel:
    def __init__(self):
        self.last_kwargs = None
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def get_conditioning_latents(self, audio_path=None):
        return ("gpt_latent", "speaker_embed")

    def _enter(self):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)

    def _exit(self):
        with self._lock:
            self.concurrent -= 1

    def inference(self, **kwargs):
        self.last_kwargs = kwargs
        self._enter()
        time.sleep(0.05)
        self._exit()
        return {"wav": np.zeros(2400, dtype=np.float32)}


def _fresh_engine(server_module):
    engine = server_module.TTSEngine()
    fake = FakeModel()
    engine.model = fake
    engine.speaker_latents["mike"] = ("gpt_latent", "speaker_embed")
    return engine, fake


def test_synthesize_passes_prosody_controls(server_module):
    engine, fake = _fresh_engine(server_module)
    engine.synthesize("Hello there.", voice="mike", language="en")
    kw = fake.last_kwargs
    assert kw is not None
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "length_penalty",
        "speed",
    ):
        assert key in kw, f"prosody control {key!r} not passed to model.inference"
    assert kw.get("enable_text_splitting") is True


def test_synthesize_accepts_per_request_overrides(server_module):
    engine, fake = _fresh_engine(server_module)
    engine.synthesize(
        "Override please.",
        voice="mike",
        language="en",
        speed=1.2,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=3.0,
    )
    kw = fake.last_kwargs
    assert kw["speed"] == 1.2
    assert kw["temperature"] == 0.5
    assert kw["top_p"] == 0.9
    assert kw["repetition_penalty"] == 3.0


def test_engine_has_inference_lock(server_module):
    engine, _ = _fresh_engine(server_module)
    assert hasattr(engine, "_infer_lock"), "engine must expose an inference lock"


def test_concurrent_synthesis_is_serialized(server_module):
    engine, fake = _fresh_engine(server_module)

    def worker(i):
        # Distinct text per call so the LRU cache never short-circuits inference.
        engine.synthesize(f"utterance number {i}", voice="mike", language="en")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake.max_concurrent == 1, (
        f"inference ran {fake.max_concurrent} calls concurrently; the inference "
        "lock must serialize access to the shared XTTS model"
    )
