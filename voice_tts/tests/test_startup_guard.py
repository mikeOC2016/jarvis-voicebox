"""
Tests for §5.1 startup guard in voice_tts/server.py.

Per ADAPTER_DESIGN_PHASE1A.md §5.1:
- TTS_API_KEY required by default on every bind interface.
- Only explicit ALLOW_UNAUTH_TTS=1 escapes the requirement.
- Refuse to start with non-zero exit + exact ERROR log when key unset and
  escape hatch not set.
- Emit exact WARNING log and continue when escape hatch is explicit.
- Guard runs BEFORE the XTTS model is loaded (no GPU allocation).

Implementation strategy: spawn a subprocess that stubs torchaudio (so server.py
imports without the real torchaudio package) and imports server. The startup
guard runs at module import time. We assert exit code and the exact log
strings printed to stderr.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_VOICE_TTS_DIR = Path(__file__).resolve().parent.parent

ERROR_LINE = (
    "[TTS-STARTUP] refusing to start: "
    "TTS_API_KEY is unset and ALLOW_UNAUTH_TTS!=1"
)
WARNING_LINE = (
    "[TTS-STARTUP] WARNING: running with unauthenticated /speak; "
    "ALLOW_UNAUTH_TTS=1 set explicitly"
)


def _run_import_server(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    """Spawn a python subprocess that stubs torchaudio and imports server.

    Captures stdout/stderr/returncode. The subprocess inherits the parent
    env (so user site-packages like numpy resolve) but TTS_API_KEY and
    ALLOW_UNAUTH_TTS are first scrubbed and then re-applied from
    env_overrides, so the test is not influenced by a developer shell.
    """
    script = textwrap.dedent(
        """
        import sys, types, os

        # Stub torchaudio so server.py's module-level import and .load monkeypatch work
        # on hosts where the package is not installed. This stub never loads audio;
        # if anything ever calls .load() the subprocess fails loudly.
        stub = types.ModuleType('torchaudio')
        def _stub_load(*a, **k):
            raise RuntimeError('torchaudio stub: load() called in test')
        stub.load = _stub_load
        sys.modules['torchaudio'] = stub

        # Sanity: TTS must NOT be imported before or during this import.
        # The startup guard MUST fire before engine.load() touches XTTS.
        assert 'TTS' not in sys.modules, 'TTS imported before guard ran'

        try:
            import server  # noqa: F401
        except SystemExit as e:
            print(f'EXIT_CODE={int(e.code) if e.code is not None else 0}')
            sys.exit(int(e.code) if e.code is not None else 0)

        # If we got here, import succeeded without sys.exit.
        # Confirm engine.load() has NOT run (XTTS not imported).
        assert 'TTS' not in sys.modules, 'TTS imported during server import'
        print('IMPORT_OK')
        """
    ).strip()

    import os
    env = dict(os.environ)
    # Scrub auth env vars so the test is not influenced by a developer shell.
    env.pop("TTS_API_KEY", None)
    env.pop("ALLOW_UNAUTH_TTS", None)
    env["PYTHONPATH"] = str(_VOICE_TTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("LOG_LEVEL", "INFO")
    env.update(env_overrides)

    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_refuses_to_start_when_key_unset_and_escape_not_set():
    """TTS_API_KEY="" and ALLOW_UNAUTH_TTS="" -> non-zero exit + exact ERROR line."""
    result = _run_import_server({"TTS_API_KEY": "", "ALLOW_UNAUTH_TTS": ""})

    assert result.returncode != 0, (
        f"expected non-zero exit; got 0. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "IMPORT_OK" not in result.stdout, "server should have refused to start"
    assert ERROR_LINE in result.stderr, (
        f"missing exact ERROR line in stderr.\n"
        f"expected: {ERROR_LINE!r}\nstderr: {result.stderr!r}"
    )


def test_refuses_when_both_env_vars_completely_unset():
    """Same as above but with the env vars entirely absent from environ."""
    # Pop them out so they're not even set to empty.
    result = _run_import_server({})
    assert result.returncode != 0
    assert ERROR_LINE in result.stderr


def test_warns_and_continues_when_escape_hatch_set_explicitly():
    """TTS_API_KEY="" and ALLOW_UNAUTH_TTS="1" -> import OK + exact WARNING line."""
    result = _run_import_server({"TTS_API_KEY": "", "ALLOW_UNAUTH_TTS": "1"})

    assert result.returncode == 0, (
        f"expected exit 0; got {result.returncode}. stderr={result.stderr!r}"
    )
    assert "IMPORT_OK" in result.stdout
    assert WARNING_LINE in result.stderr, (
        f"missing exact WARNING line in stderr.\n"
        f"expected: {WARNING_LINE!r}\nstderr: {result.stderr!r}"
    )
    assert ERROR_LINE not in result.stderr, "ERROR line should NOT appear when escape hatch is set"


def test_starts_cleanly_when_key_set_regardless_of_escape_hatch():
    """TTS_API_KEY="anything" -> import OK, no startup ERROR/WARNING line."""
    for allow in ("", "1", "0", "yes"):
        result = _run_import_server(
            {"TTS_API_KEY": "validkeyforthistest", "ALLOW_UNAUTH_TTS": allow}
        )
        assert result.returncode == 0, (
            f"expected exit 0 for ALLOW_UNAUTH_TTS={allow!r}; "
            f"got {result.returncode}. stderr={result.stderr!r}"
        )
        assert "IMPORT_OK" in result.stdout
        assert ERROR_LINE not in result.stderr
        assert WARNING_LINE not in result.stderr, (
            f"WARNING should NOT appear when TTS_API_KEY is set "
            f"(ALLOW_UNAUTH_TTS={allow!r}). stderr={result.stderr!r}"
        )


def test_guard_runs_before_xtts_model_load():
    """No GPU/XTTS allocation when the guard refuses startup.

    We assert this indirectly: the subprocess script checks that 'TTS' is
    not in sys.modules at either the catch site or after a clean import.
    Refused-startup runs never reach engine.load() because sys.exit halts
    before the FastAPI startup event fires (the event handler is registered
    but only triggered by uvicorn/TestClient, not by bare `import server`).
    """
    # Refused-startup path: TTS must never load.
    result = _run_import_server({"TTS_API_KEY": "", "ALLOW_UNAUTH_TTS": ""})
    assert result.returncode != 0
    # No noisy "TTS imported" assertion error in the subprocess output.
    assert "TTS imported" not in result.stderr
    assert "TTS imported" not in result.stdout

    # Clean-import path with escape hatch: TTS still must not load
    # because engine.load() is wired to the FastAPI startup event,
    # which is not fired by bare import.
    result = _run_import_server({"TTS_API_KEY": "", "ALLOW_UNAUTH_TTS": "1"})
    assert result.returncode == 0
    assert "TTS imported" not in result.stderr
    assert "TTS imported" not in result.stdout
