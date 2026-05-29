"""Speech-oriented text normalization for XTTS-v2.

XTTS-v2 handles plain prose better than raw operational text with IDs, symbols,
all-caps acronyms, and phone-style numbers. Keep this deterministic and local:
no network calls, no ML, no pronunciation guesses beyond the explicit lexicon.
"""
from __future__ import annotations

import re

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_SMALL = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ACRONYMS = {
    "APEX": "Apex",
    "CFO": "C F O",
    "JAX": "Jax",
    "PGX": "P G X",
    "TWS": "T W S",
    "IBKR": "I B K R",
    "DQ": "D Q",
    "POS": "P O S",
    "QOZP": "Q O Z P",
    "API": "A P I",
    "GPU": "G P U",
}


def _under_1000(n: int) -> str:
    if n < 20:
        return _SMALL[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_SMALL[ones]}"
    hundreds, rem = divmod(n, 100)
    return f"{_SMALL[hundreds]} hundred" if rem == 0 else f"{_SMALL[hundreds]} hundred {_under_1000(rem)}"


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(abs(n))
    if n < 1000:
        return _under_1000(n)
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        prefix = _int_to_words(thousands) + " thousand"
        return prefix if rem == 0 else prefix + " " + _under_1000(rem)
    millions, rem = divmod(n, 1_000_000)
    prefix = _int_to_words(millions) + " million"
    return prefix if rem == 0 else prefix + " " + _int_to_words(rem)


def _digits_to_words(s: str, sep: str = " ") -> str:
    return sep.join(_DIGIT_WORDS[ch] for ch in s if ch.isdigit())


def _expand_currency(match: re.Match[str]) -> str:
    dollars_raw, cents_raw = match.group(1), match.group(2)
    dollars = int(dollars_raw.replace(",", ""))
    words = _int_to_words(dollars) + (" dollar" if dollars == 1 else " dollars")
    if cents_raw is not None:
        cents = int(cents_raw[:2].ljust(2, "0"))
        if cents:
            words += " and " + _int_to_words(cents) + (" cent" if cents == 1 else " cents")
    return words


def _expand_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = " ".join(match.group(3).upper())
    if minute == 0:
        return f"{_int_to_words(hour)} o clock {suffix}"
    if minute < 10:
        minute_words = "oh " + _int_to_words(minute)
    else:
        minute_words = _int_to_words(minute)
    return f"{_int_to_words(hour)} {minute_words} {suffix}"


def _expand_phone(match: re.Match[str]) -> str:
    return ", ".join(_digits_to_words(part) for part in match.groups())


def _expand_number(match: re.Match[str]) -> str:
    token = match.group(0)
    clean = token.replace(",", "")
    if "." in clean:
        left, right = clean.split(".", 1)
        return _int_to_words(int(left)) + " point " + _digits_to_words(right)
    if "," in token:
        return _int_to_words(int(clean))
    if len(clean) >= 4:
        return _digits_to_words(clean)
    return _int_to_words(int(clean))


def normalize_tts_text(text: str) -> str:
    """Return text shaped for more reliable XTTS pronunciation."""
    out = str(text)
    out = out.replace("&", " and ")
    out = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", out)

    for source, spoken in _ACRONYMS.items():
        out = re.sub(rf"\b{re.escape(source)}\b", spoken, out)

    out = re.sub(r"\$(\d[\d,]*)(?:\.(\d{1,2}))?", _expand_currency, out)
    out = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp][Mm])\b", _expand_time, out)
    out = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", _expand_phone, out)
    out = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", _expand_number, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out.strip()
