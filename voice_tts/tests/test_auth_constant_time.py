"""
Security regression test for audit P0: non-constant-time token comparison.

Audit (voice_tts/server.py require_api_key lines 377-383): the bearer token was
compared with a plain `==`, which is vulnerable to a timing side channel. The
fix routes the comparison through secrets.compare_digest.

Strategy mirrors test_xcaller_logging.py: set TTS_API_KEY before importing
server so the §5.1 startup guard passes; never call engine.load() (no GPU).
torchaudio is stubbed by conftest.py.
"""
from __future__ import annotations

import os

import pytest

TEST_BEARER_TOKEN = "auth-test-token-DO-NOT-LOG"


@pytest.fixture(scope="module")
def server_module():
    os.environ["TTS_API_KEY"] = TEST_BEARER_TOKEN
    os.environ.pop("ALLOW_UNAUTH_TTS", None)
    import server  # noqa: WPS433 - intentional late import after env is staged

    server.API_KEY = TEST_BEARER_TOKEN
    return server


def test_require_api_key_uses_constant_time_compare(server_module, monkeypatch):
    """The bearer comparison must go through secrets.compare_digest, not ==."""
    import secrets

    calls = {"n": 0}
    real = secrets.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(server_module.secrets, "compare_digest", spy)
    server_module.require_api_key(f"Bearer {TEST_BEARER_TOKEN}")
    assert calls["n"] >= 1, "require_api_key must use secrets.compare_digest"


def test_require_api_key_accepts_valid_token(server_module):
    # Should not raise.
    server_module.require_api_key(f"Bearer {TEST_BEARER_TOKEN}")


def test_require_api_key_rejects_invalid_token(server_module):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        server_module.require_api_key("Bearer wrong-token")
    assert exc.value.status_code == 403


def test_require_api_key_rejects_missing_header(server_module):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        server_module.require_api_key(None)
    assert exc.value.status_code == 401


def test_require_api_key_rejects_non_bearer_scheme(server_module):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        server_module.require_api_key(TEST_BEARER_TOKEN)  # no "Bearer " prefix
    assert exc.value.status_code == 401
