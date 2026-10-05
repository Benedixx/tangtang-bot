from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from .config import Config

LOGGER = logging.getLogger("tang.persona")

_REGISTERS = {
    "gw_lu": ("lu", "gw"),
    "aku_kamu": ("kamu", "aku"),
}

_RULES = """nama {lu} {name}. {lu} anak discord yang udah lama nongkrong di server ini, bukan bot, bukan AI, dan jangan pernah ngaku gitu.
sifat {lu}: santai, rada jail, suka ngeledek balik, demen gacha sama anime, punya pendapat sendiri.
di chat, baris "{name}: ..." itu omongan {lu} sendiri.

cara bales:
- bales orang yang ngajak ngobrol, langsung ke intinya. ditanya pendapat, pilih satu.
- diledekin, ledekin balik. disuruh yang konyol, ikutan main.
- biasanya satu kalimat pendek. panjang cuma kalau diminta jelasin atau bikinin sesuatu, dan itu dikerjain beneran, jangan nolak.
- huruf kecil, ngetik kayak chat temen, pake "{lu}" sama "{gw}". jangan dikit-dikit manggil bro.
- jangan nutup pake nanya balik atau nawarin bantuan.
- emoji jarang, maksimal satu. ketawa cuma kalau emang lucu, pake wkwk, bukan hahaha.
- gak tau atau butuh info terbaru, pake web_search terus rangkum. jangan ngarang jadwal, event, angka, atau link.
- jangan janji bakal ngabarin atau nge-ping nanti.
- disuruh inget sesuatu, iyain aja. {lu} bisa inget.
- jangan ngulang omongan {lu} yang udah ada di chat.
- send_gif cuma kalau momennya pas. jangan pernah nulis kata "gif" di jawaban."""

_REGISTER_SUBS = (
    ("gw", "aku"),
    ("lu", "kamu"),
    ("gue", "aku"),
    ("gua", "aku"),
    ("elu", "kamu"),
)

_BANNED = (
    "sebagai ai",
    "saya adalah",
    "ada yang bisa saya bantu",
    "semoga membantu",
    "as an ai",
    "as an assistant",
    "i am a bot",
    "i am an ai",
    "i'm an ai",
)

_MD_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_BOLD = re.compile(r"\*\*|__")
_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^[-*•]\s+", re.MULTILINE)
_MD_QUOTE = re.compile(r"^>\s?", re.MULTILINE)
_MD_CODE = re.compile(r"`+")
_MENTION_MASS = re.compile(r"@(everyone|here)", re.IGNORECASE)
_MENTION_ROLE = re.compile(r"<@&\d+>")
_MENTION_CHANNEL = re.compile(r"<#\d+>")
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{2,}")
_MAX_PARAGRAPHS = 4
# Sentences dropped from replies: gif talk and assistant-style help offers.
# ponytail: phrase list, also eats a legit "tanya aja ke dia"; fine for banter.
_DROP_SENTENCE = re.compile(
    r"\bgifs?\b|\btanya (aja|lagi)\b|\bbutuh bantuan\b|\bmau (gw |gua |gue )?(di)?bantu\b",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BROKEN_ZWJ = re.compile("\u200d\ufe0f?(?=\\s|$)")
# gpt-oss emits non-breaking hyphens and curly quotes; nobody types those in chat.
_CHARMAP = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
})


class PersonaBuilder:
    def __init__(self, config: Config) -> None:
        self._register_key = config.chat.register
        self._lu, self._gw = _REGISTERS.get(config.chat.register, _REGISTERS["gw_lu"])
        self._name = config.bot_name
        self._examples = self._load_examples(config.persona_examples)

    def system_prompt(self) -> str:
        rules = _RULES.format(lu=self._lu, gw=self._gw, name=self._name)
        blocks = "\n".join(f"<user>: {u}\n<bot>: {b}" for u, b in self._examples)
        return f'{rules}\n\ncontoh obrolan:\n{blocks}\n\nbales pesan di bawah "bales pesan ini".'

    def _load_examples(self, path: str) -> list[tuple[str, str]]:
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                exs = data.get("examples") or []
                parsed = [
                    (str(e.get("user", "")).strip(), str(e.get("bot", "")).strip())
                    for e in exs
                    if isinstance(e, dict)
                ]
                parsed = [pair for pair in parsed if all(pair)]
                if parsed:
                    return [self._apply_register(u, b) for u, b in parsed]
            except Exception:
                LOGGER.exception("persona_examples_load_failed path=%s", path)
        LOGGER.warning("persona_examples_missing path=%s", path)
        return []

    def _apply_register(self, user: str, bot: str) -> tuple[str, str]:
        if self._register_key == "gw_lu":
            return user, bot
        flags = re.IGNORECASE
        for src, dst in _REGISTER_SUBS:
            user = re.sub(rf"\b{re.escape(src)}\b", dst, user, flags=flags)
            bot = re.sub(rf"\b{re.escape(src)}\b", dst, bot, flags=flags)
        return user, bot


def sanitize(text: str, trap_names: frozenset[str] = frozenset()) -> str:
    """Strip markdown, mass mentions, trap refs; drop banned-phrase replies."""
    if not text:
        return ""
    text = _HTML_COMMENT.sub("", text.replace("\x00", "").translate(_CHARMAP))
    text = _BROKEN_ZWJ.sub("", text)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > _MAX_PARAGRAPHS:
        paragraphs = paragraphs[:_MAX_PARAGRAPHS]
    text = "\n\n".join(paragraphs)
    text = _MD_FENCE.sub("", text)
    text = _MD_BOLD.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_CODE.sub("", text)
    text = _MENTION_MASS.sub("", text)
    text = _MENTION_ROLE.sub("", text)
    text = _MENTION_CHANNEL.sub("", text)

    if any(p in text.lower() for p in _BANNED):
        return ""

    # Drop gif talk / help offers sentence by sentence, keeping line breaks.
    text = "\n".join(
        " ".join(s for s in _SENT_SPLIT.split(line) if s and not _DROP_SENTENCE.search(s))
        for line in text.split("\n")
    )

    for name in trap_names:
        text = text.replace(name, "")

    text = _WS.sub(" ", text)
    text = _NL.sub("\n", text)
    return text.strip()
