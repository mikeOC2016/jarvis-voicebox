"""
Tests for §5.2 X-Caller logging in voice_tts/server.py.

Per ADAPTER_DESIGN_PHASE1A.md §5.2:
- Allowlist (case-sensitive, exact match): ["jax", "cfo-dashboard-events",
  "dashboard-rag", "smoke", "hub"].
- Non-allowlisted / empty / missing X-Caller -> "unknown" (request not rejected).
- Every /speak and /speak/stream call emits ONE log line with EXACTLY
  the format: "[TTS] caller=<sanitized> status=<int> latency_ms=<n>".
- Spoken text body and bearer token MUST NEVER appear in any log output.

Implementation strategy: pre-set TTS_API_KEY before importing server so the
§5.1 startup guard passes. Monkeypatch engine.synthesize/synthesize_stream
to skip XTTS so no GPU/model load occurs. Drive the HTTP layer with
FastAPI's TestClient and assert against pytest's caplog.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

import numpy as np
import pytest

# Bearer token used by all happy-path requests in this file. Set BEFORE
# importing server so the §5.1 startup guard sees a non-empty TTS_API_KEY
# at module import time. Distinctive value so we can assert it never leaks
# into any log record.
TEST_BEARER_TOKEN = "xcaller-test-token-DO-NOT-LOG-this-string"

# Distinctive spoken text bodies used in tests. We assert none of these
# strings ever appears in any captured log record.
SECRET_TEXT_SHORT = "SECRETSPEECHTOKEN_alpha"
SECRET_TEXT_LONG = "SECRETSPEECHTOKEN_bravo_padded_with_more_words_so_the_payload_is_long"


def _install_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", TEST_BEARER_TOKEN)
    monkeypatch.delenv("ALLOW_UNAUTH_TTS", raising=False)


@pytest.fixture(scope="module")
def server_module():
    """Import server once per module with TTS_API_KEY set so the startup guard passes."""
    import os
    os.environ["TTS_API_KEY"] = TEST_BEARER_TOKEN
    os.environ.pop("ALLOW_UNAUTH_TTS", None)
    import server  # noqa: WPS433 - intentional late import after env is staged
    # Force-sync the module-level API_KEY in case server was imported earlier
    # in the test session with a different env. Re-importing is not enough
    # because server.API_KEY is a module-level binding evaluated once.
    server.API_KEY = TEST_BEARER_TOKEN
    return server


@pytest.fixture()
def client(server_module, monkeypatch):
    """TestClient with engine synthesis mocked (no XTTS, no GPU)."""
    fake_audio = np.zeros(2400, dtype=np.float32)  # 0.1 s of silence at 24 kHz
    sample_rate = 24000

    def fake_synth(text: str, voice: str, language: str, **kwargs):
        return fake_audio, sample_rate

    def fake_stream(text: str, voice: str, language: str, **kwargs) -> Iterator[bytes]:
        # Yield a single tiny WAV-ish chunk; the test does not parse audio.
        yield b"\x00\x00\x00\x00"

    monkeypatch.setattr(server_module.engine, "synthesize", fake_synth)
    monkeypatch.setattr(server_module.engine, "synthesize_stream", fake_stream)
    # The synthesize methods check engine.model; bypass that.
    monkeypatch.setattr(server_module.engine, "model", object())

    from fastapi.testclient import TestClient
    return TestClient(server_module.app)


def _tts_log_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return only the [TTS] structured log messages."""
    return [r.message for r in caplog.records if r.message.startswith("[TTS]")]


def _all_log_text(caplog: pytest.LogCaptureFixture) -> str:
    """Return the full captured log text (messages + formatted strings)."""
    parts: list[str] = []
    for r in caplog.records:
        parts.append(r.getMessage())
        # Args may carry data that didn't get into the formatted message.
        if r.args:
            try:
                parts.append(repr(r.args))
            except Exception:
                pass
    return "\n".join(parts)


LINE_RE = re.compile(r"^\[TTS\] caller=(?P<caller>[A-Za-z0-9_\-]+) status=(?P<status>\d+) latency_ms=(?P<latency>\d+)$")


# --- Allowlist coverage ----------------------------------------------------

@pytest.mark.parametrize(
    "header_value",
    ["jax", "cfo-dashboard-events", "dashboard-rag", "smoke", "hub"],
)
def test_allowlisted_callers_pass_through(client, caplog, header_value):
    caplog.set_level(logging.INFO, logger="voice_tts")
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": header_value,
        },
    )
    assert resp.status_code == 200

    lines = _tts_log_lines(caplog)
    assert len(lines) == 1, f"expected exactly one [TTS] line, got: {lines}"
    m = LINE_RE.match(lines[0])
    assert m, f"line {lines[0]!r} does not match exact format"
    assert m.group("caller") == header_value
    assert m.group("status") == "200"


@pytest.mark.parametrize(
    "header_value",
    [
        "JAX",                  # case-sensitive: uppercase is not allowlisted
        "dashboard-events",     # retired round-1 name
        "hub-events",           # retired round-1 name
        "attacker-string",
        "jax ",                 # trailing space
        " jax",                 # leading space
        "jax;dropall",
        "<script>",
        "",                     # empty
    ],
)
def test_non_allowlisted_normalizes_to_unknown(client, caplog, header_value):
    caplog.set_level(logging.INFO, logger="voice_tts")
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": header_value,
        },
    )
    assert resp.status_code == 200

    lines = _tts_log_lines(caplog)
    assert len(lines) == 1
    m = LINE_RE.match(lines[0])
    assert m, f"line {lines[0]!r} does not match exact format"
    assert m.group("caller") == "unknown", (
        f"header {header_value!r} must normalize to 'unknown', got {m.group('caller')!r}"
    )


def test_missing_xcaller_header_normalizes_to_unknown(client, caplog):
    caplog.set_level(logging.INFO, logger="voice_tts")
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
    )
    assert resp.status_code == 200

    lines = _tts_log_lines(caplog)
    assert len(lines) == 1
    m = LINE_RE.match(lines[0])
    assert m
    assert m.group("caller") == "unknown"


def test_stream_endpoint_also_logs_with_same_format(client, caplog):
    caplog.set_level(logging.INFO, logger="voice_tts")
    resp = client.post(
        "/speak/stream",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": "hub",
        },
    )
    # Drain the streaming body so the response cycle (and middleware) completes.
    _ = resp.content
    assert resp.status_code == 200

    lines = _tts_log_lines(caplog)
    assert len(lines) == 1
    m = LINE_RE.match(lines[0])
    assert m, f"streaming line {lines[0]!r} does not match exact format"
    assert m.group("caller") == "hub"
    assert m.group("status") == "200"


# --- Format invariants -----------------------------------------------------

def test_log_line_format_exact_match(client, caplog):
    """Format MUST be exactly `[TTS] caller=<s> status=<int> latency_ms=<int>`."""
    caplog.set_level(logging.INFO, logger="voice_tts")
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": "smoke",
        },
    )
    assert resp.status_code == 200
    lines = _tts_log_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert LINE_RE.match(line), (
        f"expected exact format '[TTS] caller=<s> status=<int> latency_ms=<int>', got {line!r}"
    )
    # latency_ms is a non-negative integer (no float, no unit suffix)
    m = LINE_RE.match(line)
    assert int(m.group("latency")) >= 0


# --- No leakage of bodies or bearer tokens --------------------------------

def test_spoken_text_body_does_not_appear_in_any_log(client, caplog):
    caplog.set_level(logging.DEBUG)  # widest level so nothing is filtered out
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_LONG, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": "jax",
        },
    )
    assert resp.status_code == 200
    full = _all_log_text(caplog)
    assert SECRET_TEXT_LONG not in full, (
        f"spoken text leaked into logs:\n{full}"
    )


def test_bearer_token_does_not_appear_in_any_log(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/speak",
        json={"text": SECRET_TEXT_SHORT, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": "jax",
        },
    )
    assert resp.status_code == 200
    full = _all_log_text(caplog)
    assert TEST_BEARER_TOKEN not in full, "bearer token leaked into logs"
    # Also assert the literal "Bearer " token form does not appear.
    assert f"Bearer {TEST_BEARER_TOKEN}" not in full


def test_stream_endpoint_does_not_leak_text_or_token(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/speak/stream",
        json={"text": SECRET_TEXT_LONG, "voice": "mike", "language": "en"},
        headers={
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "X-Caller": "cfo-dashboard-events",
        },
    )
    _ = resp.content
    assert resp.status_code == 200
    full = _all_log_text(caplog)
    assert SECRET_TEXT_LONG not in full
    assert TEST_BEARER_TOKEN not in full
