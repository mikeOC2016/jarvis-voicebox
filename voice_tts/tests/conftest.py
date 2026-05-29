"""
Shared pytest fixtures and import-time stubs for voice_tts tests.

Tests run on the PGX host where torchaudio is not installed (it lives only
inside the voice_tts Docker container). We stub torchaudio at sys.modules
level BEFORE server.py is imported so its module-level
`import torchaudio` and `torchaudio.load = ...` monkeypatch succeed without
needing the real package. The heavy XTTS model load happens inside
engine.load() which tests never call, so no GPU/model allocation occurs.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_VOICE_TTS_DIR = Path(__file__).resolve().parent.parent
if str(_VOICE_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(_VOICE_TTS_DIR))

if "torchaudio" not in sys.modules:
    _stub = types.ModuleType("torchaudio")

    def _stub_load(*args, **kwargs):  # pragma: no cover - never called in tests
        raise RuntimeError("torchaudio stub: load() should not run in tests")

    _stub.load = _stub_load
    sys.modules["torchaudio"] = _stub
