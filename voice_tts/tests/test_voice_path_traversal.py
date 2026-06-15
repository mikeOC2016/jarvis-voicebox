"""
Security regression test for audit P0: path traversal in voice_path().

Audit (voice_tts/server.py voice_path lines 256-260): `voice` allowed path
traversal via `../name`, so a crafted voice could escape VOICES_DIR. The fix
confines the resolved path to VOICES_DIR and rejects separators/traversal.

Strategy mirrors test_xcaller_logging.py: set TTS_API_KEY before importing
server so the §5.1 startup guard passes; never call engine.load() (no GPU).
torchaudio is stubbed by conftest.py.
"""
from __future__ import annotations

import os

import pytest

TEST_BEARER_TOKEN = "traversal-test-token-DO-NOT-LOG"


@pytest.fixture(scope="module")
def server_module():
    os.environ["TTS_API_KEY"] = TEST_BEARER_TOKEN
    os.environ.pop("ALLOW_UNAUTH_TTS", None)
    import server  # noqa: WPS433 - intentional late import after env is staged

    server.API_KEY = TEST_BEARER_TOKEN
    return server


@pytest.fixture()
def voices_dir(tmp_path, server_module, monkeypatch):
    """A temp voices dir with one legit voice and a secret file just outside it."""
    vdir = tmp_path / "voices"
    vdir.mkdir()
    (vdir / "mike.wav").write_bytes(b"RIFF....WAVEfake")
    # A wav sitting one level above the voices dir; traversal must NOT reach it.
    (tmp_path / "secret.wav").write_bytes(b"RIFF....WAVEsecret")
    monkeypatch.setattr(server_module, "VOICES_DIR", vdir)
    return vdir


def test_voice_path_returns_legit_voice(server_module, voices_dir):
    engine = server_module.TTSEngine()
    p = engine.voice_path("mike")
    assert p == (voices_dir / "mike.wav").resolve()
    assert p.exists()


@pytest.mark.parametrize(
    "evil",
    [
        "../secret",
        "../../etc/passwd",
        "sub/mike",
        "/etc/passwd",
        "mike/../../secret",
        "..",
        ".",
        "",
        "mike\x00",
    ],
)
def test_voice_path_rejects_traversal(server_module, voices_dir, evil):
    from fastapi import HTTPException

    engine = server_module.TTSEngine()
    with pytest.raises(HTTPException) as exc:
        engine.voice_path(evil)
    # Traversal/invalid names are a client error, never a 200 that serves the file.
    assert exc.value.status_code in (400, 404)


def test_voice_path_cannot_reach_file_outside_voices_dir(server_module, voices_dir):
    """`../secret` must not resolve to the real secret.wav one level up."""
    from fastapi import HTTPException

    engine = server_module.TTSEngine()
    with pytest.raises(HTTPException):
        engine.voice_path("../secret")
