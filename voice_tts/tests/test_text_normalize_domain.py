"""
Tests for audit P1 #6: text_normalize mangling domain tokens.

Audit (text_normalize.py): percents, signed moves, dates, and suffix-decimal /
alphanumeric tokens were mangled into unnatural token streams XTTS reads flat or
wrong. Required behaviors (per codex round-3 verdict + user brief examples
+1.25%, 0DTE, dates):

  - "%"            -> "percent"
  - signed numbers -> "plus"/"minus"
  - MM/DD/YYYY     -> spoken date (no slashes, month name, ordinal day, year)
  - suffix tokens  -> digits spoken, not left raw or half-mangled (22.4x, 0DTE, 10bps, 3M)
  - id-with-digits -> digits spelled (DU3317391)
  - None / empty   -> ""
"""
from __future__ import annotations

from text_normalize import normalize_tts_text


# --- percents and signed moves --------------------------------------------


def test_signed_percent_up():
    assert normalize_tts_text("+1.25%") == "plus one point two five percent"


def test_signed_percent_down():
    assert normalize_tts_text("-3.5%") == "minus three point five percent"


def test_plain_percent():
    assert normalize_tts_text("Margins at 42%.") == "Margins at forty two percent."


def test_percent_symbol_never_survives():
    out = normalize_tts_text("up 1.25% today")
    assert "%" not in out
    assert "percent" in out


# --- dates -----------------------------------------------------------------


def test_date_is_spoken_not_slashed():
    out = normalize_tts_text("Filed 05/29/2026.")
    assert "/" not in out
    low = out.lower()
    assert "may" in low
    assert "twenty ninth" in low
    assert "two thousand twenty six" in low


# --- suffix-decimal / alphanumeric tokens ---------------------------------


def test_leverage_suffix_decimal():
    out = normalize_tts_text("trading at 22.4x")
    assert "twenty two point four" in out
    assert ".4x" not in out
    assert "22" not in out


def test_zero_dte_token():
    out = normalize_tts_text("0DTE expiry")
    assert "zero" in out
    assert "0DTE" not in out
    assert "0" not in out  # the digit must be spoken, not left raw


def test_bps_token_digits_spoken():
    out = normalize_tts_text("10bps move")
    assert "ten" in out
    assert "10bps" not in out


def test_account_id_digits_spelled():
    out = normalize_tts_text("IBKR DU3317391")
    assert "I B K R" in out
    assert "DU3317391" not in out
    # digits spelled out individually
    assert "three three one seven three nine one" in out


# --- empty / None ----------------------------------------------------------


def test_none_returns_empty_string():
    assert normalize_tts_text(None) == ""


def test_empty_returns_empty_string():
    assert normalize_tts_text("") == ""


# --- regression: currency and sentence punctuation still intact -----------


def test_currency_still_expands():
    out = normalize_tts_text("Total $1,234.56 due.")
    assert out == "Total one thousand two hundred thirty four dollars and fifty six cents due."


def test_sentence_punctuation_preserved():
    out = normalize_tts_text("First sentence. Second one? Third one!")
    assert out == "First sentence. Second one? Third one!"
