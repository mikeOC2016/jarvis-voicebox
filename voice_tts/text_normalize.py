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


_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_ORD_ONES = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth",
}
_ORD_TENS = {20: "twentieth", 30: "thirtieth"}


def _speak_number(tok: str) -> str:
    """Speak a numeric token the way _expand_number does, from a bare string."""
    clean = tok.replace(",", "")
    if "." in clean:
        left, right = clean.split(".", 1)
        left_words = _int_to_words(int(left)) if left else "zero"
        return left_words + " point " + _digits_to_words(right)
    if "," in tok:
        return _int_to_words(int(clean))
    if len(clean) >= 4:
        return _digits_to_words(clean)
    return _int_to_words(int(clean))


def _spell_letters(s: str) -> str:
    """Spell a run of letters: 'DTE' -> 'D T E', single letter unchanged."""
    return s if len(s) == 1 else " ".join(s)


def _ordinal_day(n: int) -> str:
    if 1 <= n < 20:
        return _ORD_ONES[n]
    if n in _ORD_TENS:
        return _ORD_TENS[n]
    tens, ones = divmod(n, 10)
    return f"{_TENS[tens]} {_ORD_ONES[ones]}"


def _expand_number(match: re.Match[str]) -> str:
    return _speak_number(match.group(0))


def _expand_percent(match: re.Match[str]) -> str:
    sign = match.group(1)
    prefix = "plus " if sign == "+" else "minus " if sign == "-" else ""
    return prefix + _speak_number(match.group(2)) + " percent"


def _expand_date(match: re.Match[str]) -> str:
    month, day, year_raw = int(match.group(1)), int(match.group(2)), match.group(3)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    return f"{_MONTHS[month]} {_ordinal_day(day)}, {_int_to_words(int(year_raw))}"


def _expand_num_suffix(match: re.Match[str]) -> str:
    """Digits-then-letters: '22.4x' -> 'twenty two point four x', '0DTE' -> 'zero D T E'."""
    return _speak_number(match.group(1)) + " " + _spell_letters(match.group(2))


def _expand_alpha_id(match: re.Match[str]) -> str:
    """Letters-then-digits id: 'DU3317391' -> 'D U three three one ...'."""
    return _spell_letters(match.group(1)) + " " + _digits_to_words(match.group(2))


def _expand_signed(match: re.Match[str]) -> str:
    sign = "plus " if match.group(1) == "+" else "minus "
    return sign + _speak_number(match.group(2))


def normalize_tts_text(text: str) -> str:
    """Return text shaped for more reliable XTTS pronunciation."""
    if text is None:
        return ""
    out = str(text)
    out = out.replace("&", " and ")
    out = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", out)

    for source, spoken in _ACRONYMS.items():
        out = re.sub(rf"\b{re.escape(source)}\b", spoken, out)

    # Signed/unsigned percentages before any other number handling so the
    # sign and the % both speak: "+1.25%" -> "plus one point two five percent".
    out = re.sub(r"(?<![\w.])([+-]?)(\d[\d,]*(?:\.\d+)?)\s*%", _expand_percent, out)
    out = re.sub(r"\$(\d[\d,]*)(?:\.(\d{1,2}))?", _expand_currency, out)
    out = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp][Mm])\b", _expand_time, out)
    out = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", _expand_phone, out)
    out = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", _expand_date, out)
    # Alphanumeric domain tokens: digits+letters then letters+digits.
    out = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)([A-Za-z]+)\b", _expand_num_suffix, out)
    out = re.sub(r"\b([A-Za-z]+)(\d+)\b", _expand_alpha_id, out)
    # Standalone signed numbers (sign not glued to a preceding word/number).
    out = re.sub(r"(?<![\w.])([+-])(\d[\d,]*(?:\.\d+)?)", _expand_signed, out)
    out = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", _expand_number, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out.strip()
