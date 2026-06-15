"""
Security regression test for audit P0: innerHTML XSS sink in the dashboard.

Audit (dashboard/app.py JS line 664): Qdrant result fields (collection/snippet)
and social-bot fields were injected via `innerHTML`, so a poisoned snippet could
execute script in the dashboard. The fix renders untrusted fields via
createElement + textContent / text nodes (never innerHTML).

This test proves the behavior: it extracts the testable render helpers from the
rendered dashboard, runs them in Node against a fake DOM whose `innerHTML` setter
THROWS, feeds a classic XSS payload, and asserts the payload survives verbatim as
text (i.e. it was never parsed as HTML).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

_START = "/*__JARVIS_TESTABLE_RENDER_START__*/"
_END = "/*__JARVIS_TESTABLE_RENDER_END__*/"

XSS_PAYLOAD = "<img src=x onerror=alert(1)>"


def _extract_testable_render() -> str:
    import app

    html = app.render_dashboard()
    assert _START in html and _END in html, (
        "dashboard must expose sentinel-wrapped testable render helpers "
        f"({_START} ... {_END})"
    )
    return html.split(_START, 1)[1].split(_END, 1)[0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_search_results_render_escapes_untrusted_html(tmp_path):
    helpers = _extract_testable_render()
    harness = textwrap.dedent(
        """
        // Fake DOM. innerHTML setter THROWS so any unsafe sink fails loudly.
        function makeEl(tag) {
          return {
            tagName: tag, className: '', _text: '', children: [],
            set textContent(v) { this._text = String(v); },
            get textContent() { return this._text; },
            set innerHTML(v) { throw new Error('UNSAFE_innerHTML_USED: ' + v); },
            appendChild(c) { this.children.push(c); return c; },
            replaceChildren() { this.children = []; },
          };
        }
        const document = {
          createElement: makeEl,
          createTextNode(t) { return { nodeType: 3, _text: String(t),
            get textContent() { return this._text; } }; },
        };
        __HELPERS__

        // collect all text rendered into the container, recursively
        function gather(node, acc) {
          if (node == null) return acc;
          if (node.nodeType === 3) { acc.push(node._text); return acc; }
          if (typeof node._text === 'string' && node._text) acc.push(node._text);
          (node.children || []).forEach(c => gather(c, acc));
          return acc;
        }

        const list = makeEl('ul');
        renderSearchResults(list, [{ collection: 'evil', snippet: PAYLOAD }]);
        const texts = gather(list, []);
        if (!texts.includes(PAYLOAD)) {
          console.error('payload not rendered verbatim as text:', JSON.stringify(texts));
          process.exit(2);
        }
        console.log('OK_SEARCH');
        """
    ).replace("__HELPERS__", helpers).replace(
        "PAYLOAD", repr_js(XSS_PAYLOAD)
    )
    script = tmp_path / "harness.mjs"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        f"node harness failed (unsafe sink or missing helper).\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "OK_SEARCH" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_social_bot_render_escapes_untrusted_html(tmp_path):
    helpers = _extract_testable_render()
    harness = textwrap.dedent(
        """
        function makeEl(tag) {
          return {
            tagName: tag, className: '', _text: '', children: [],
            set textContent(v) { this._text = String(v); },
            get textContent() { return this._text; },
            set innerHTML(v) { throw new Error('UNSAFE_innerHTML_USED: ' + v); },
            appendChild(c) { this.children.push(c); return c; },
            replaceChildren() { this.children = []; },
          };
        }
        const document = {
          createElement: makeEl,
          createTextNode(t) { return { nodeType: 3, _text: String(t),
            get textContent() { return this._text; } }; },
        };
        __HELPERS__

        function gather(node, acc) {
          if (node == null) return acc;
          if (node.nodeType === 3) { acc.push(node._text); return acc; }
          if (typeof node._text === 'string' && node._text) acc.push(node._text);
          (node.children || []).forEach(c => gather(c, acc));
          return acc;
        }

        const container = makeEl('div');
        renderSocialBots(container, [{
          title: PAYLOAD, description: PAYLOAD, badge_label: PAYLOAD,
          schedule_state: 'x', service_state: 'x', last_run_at: null,
          last_posted_at: null, posted_count: 0, errors_24h: 0,
          latest_log_line: PAYLOAD, state: 'alive',
        }]);
        const texts = gather(container, []);
        if (!texts.includes(PAYLOAD)) {
          console.error('payload not rendered verbatim as text:', JSON.stringify(texts));
          process.exit(2);
        }
        console.log('OK_BOTS');
        """
    ).replace("__HELPERS__", helpers).replace(
        "PAYLOAD", repr_js(XSS_PAYLOAD)
    )
    script = tmp_path / "harness_bots.mjs"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        f"node harness failed (unsafe sink or missing helper).\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "OK_BOTS" in result.stdout


def repr_js(s: str) -> str:
    """A JS string literal for a Python string (JSON is a safe subset)."""
    import json

    return json.dumps(s)
