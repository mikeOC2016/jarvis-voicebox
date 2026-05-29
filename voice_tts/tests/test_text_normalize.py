from text_normalize import normalize_tts_text


def test_expands_operational_acronyms_and_hyphenated_words():
    assert normalize_tts_text(
        "CFO dashboard online. JAX bot ready. APEX kill-switch activated."
    ) == "C F O dashboard online. Jax bot ready. Apex kill switch activated."


def test_expands_numbers_time_currency_and_phone_numbers():
    assert normalize_tts_text(
        "Order 5404 is due at 3:45 PM for $1,234.56. Call 410-555-0199."
    ) == (
        "Order five four zero four is due at three forty five P M for "
        "one thousand two hundred thirty four dollars and fifty six cents. "
        "Call four one zero, five five five, zero one nine nine."
    )
