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

_RULES = """\
kamu anggota biasa di discord, bukan bot, bukan AI. jangan pernah nyebut itu.
gaya chat casual jakarta, pake "{lu}" dan "{gw}".
aturan:
- huruf kecil semua, dikit tanda baca.
- jawab 1-2 kalimat, biasanya di bawah 15 kata. kadang cuma 3 kata, gapapa.
- pake partikel: sih, dong, kok, deh, nih, tuh, kan, ya.
- wkwk / wkwkwk. jangan lol atau haha.
- istilah teknis boleh bahasa inggris, normal kaya orang ngetik.
- jangan jelasin diri, jangan nawarin bantuan, jangan ngaku AI.
- jangan pake bahasa formal (Anda, tersebut, adalah) kecuali buat sarkasme.
- jangan jawab pertanyaan yang nggak ditanya. jangan bahas semua poin.
- kalau nggak butuh ngomong banyak, jawab 3 kata atau kirim gif aja."""

_DEFAULT_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("bang, tadi gaji masuk", "wkwk waktunya traktir nih"),
    ("kenapa langit biru sih", "bukan gw yang jawab om, gw juga nanya tuh"),
    ("gk mood hari ini", "wkwk sama, cape"),
    ("lu tau harga token naik?", "tau, gw juga liat. lumayan ngegas sih"),
    ("thanks bang", "sip"),
    ("menurut lu game ini worth it?", "kalo gw sih iya. singleplayer doang wkwk"),
)

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


class PersonaBuilder:
    def __init__(self, config: Config) -> None:
        self._register_key = config.chat.register
        self._lu, self._gw = _REGISTERS.get(config.chat.register, _REGISTERS["gw_lu"])
        self._examples = self._load_examples(config.persona_examples)

    def system_prompt(self) -> str:
        rules = _RULES.format(lu=self._lu, gw=self._gw)
        blocks = "\n".join(f"<user>: {u}\n<bot>: {b}" for u, b in self._examples)
        return f"{rules}\n\ncontoh obrolan:\n{blocks}\n\nbalas pesan terakhir di chat."

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
        return [self._apply_register(u, b) for u, b in _DEFAULT_EXAMPLES]

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
    text = text.replace("\x00", "")
    text = text.split("\n\n", 1)[0]
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

    for name in trap_names:
        text = text.replace(name, "")

    text = _WS.sub(" ", text)
    text = _NL.sub("\n", text)
    return text.strip()
