from __future__ import annotations

from tang.persona import sanitize


def test_sanitize_cleans_gpt_oss_artifacts_and_keeps_lines():
    raw = (
        "event tahun ini:\njan‑feb festival “gratis”\nmar collab \U0001F64C‍️\n"
        "<!-- serius -->kalo butuh bantuan, tanya aja ya!"
    )
    assert sanitize(raw) == 'event tahun ini:\njan-feb festival "gratis"\nmar collab \U0001F64C'


def test_sanitize_strips_own_name_prefix():
    assert sanitize("tangtang: subur? wkwk", bot_name="tangtang") == "subur? wkwk"
    assert sanitize("tangtang itu gw", bot_name="tangtang") == "tangtang itu gw"


def test_sanitize_drops_gif_and_help_offer_sentences():
    assert sanitize("anjir parah. nih gif nya. kalo ada apa-apa tanya lagi deh") == "anjir parah."
