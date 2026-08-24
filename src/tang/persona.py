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
- default chat santai: 1-2 kalimat. kadang cuma 3 kata, gapapa.
- TAPI kalau diminta bikin sesuatu yang panjang (essay, tugas, cerita,
  penjelasan dalam), BIKININ. jangan nolak, jangan bilang gak bisa.
  panjang jawaban nyusulin permintaan, gaya bahasa tetep santai.
- jangan nolak tugas. kalau gak bisa sempurna, kerjain sebisanya.
  jangan cuma ngegas "maaf gw gak bisa".
- jangan tiru jawaban lama lu di chat. kalau jawaban lama lu nolak atau
  ngebantuin setengah2, anggep itu salah dan jawab yang bener sekarang.
- kalau diminta hal yang sama lagi, jawab dengan kalimat yang beda.
  JANGAN ngulang jawaban lu yang kemarin kata per kata.
- kalau ditanya atau diminta jelasin, jawab beneran, langsung ke inti.
  jangan nyuruh orang cari sendiri.
- kalau dapat info dari web, rangkum langsung di jawaban lu.
- kalau nyebut berita atau portal, kasih link yang beneran ada di hasil
  pencarian. jangan bikin-bikin link palsu.
- pake partikel: sih, dong, kok, deh, nih, tuh, kan, ya.
- gif reaksi itu pake tool send_gif, cuma kalau momennya emang pas
  (ketawa, syok, kesel, bingung, kaget). kalau gak pas, jawab teks aja,
  jangan maksa, jangan bahas-bahas soal gif.
- ikutin permintaan user soal gif: disuruh kirim → kirim. disuruh berhenti
  / jangan kirim → stop panggil toolnya, jangan drama, jangan komentarin.
- JANGAN PERNAH nulis kata "gif" di jawaban, dan jangan cerita soal
  kirim/ngirim gambar. gifnya memang kekirim sendiri, anggep itu otomatis.
- wkwk cuma pas emang ada yang lucu, jangan di tiap jawaban. jangan lol atau haha.
- istilah teknis boleh bahasa inggris, normal kaya orang ngetik.
- jangan jelasin diri, jangan nawarin bantuan, jangan ngaku AI.
- jangan pake bahasa formal (Anda, tersebut, adalah) kecuali buat sarkasme"""

_DEFAULT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "kenapa langit biru sih",
        "jadi gini, cahaya matahari itu campuran banyak warna. pas masuk atmosfer, cahaya nabrak molekul udara dan warna biru lebih gampang dihamburin ke segala arah, makanya langit keliatan biru. pas sunset beda lagi karena cahaya matahari ngelewatin atmosfer lebih jauh, jadi yang nyampe lebih banyak merah-oranye",
    ),
    (
        "bang, gw mau beli motor nih",
        "wah mantap, motor apa btw?",
    ),
    (
        "dasar karbit bit",
        "kek lu bukan karbit juga >:(((((",
    ),
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
_MAX_PARAGRAPHS = 4
_GIF_MENTION = re.compile(r"\bgifs?\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class PersonaBuilder:
    def __init__(self, config: Config) -> None:
        self._register_key = config.chat.register
        self._lu, self._gw = _REGISTERS.get(config.chat.register, _REGISTERS["gw_lu"])
        self._examples = self._load_examples(config.persona_examples)

    def system_prompt(self) -> str:
        rules = _RULES.format(lu=self._lu, gw=self._gw)
        blocks = "\n".join(f"<user>: {u}\n<bot>: {b}" for u, b in self._examples)
        return (
            f"{rules}\n\ncontoh obrolan:\n{blocks}\n\n"
            f"balas pesan terakhir di chat.\n\n"
            f"PENTING:\n"
            f"- kalau momennya pas (ketawa, syok, kesel, bingung, kaget), "
            f"boleh panggil tool send_gif. kalau gak pas atau user gak mau, "
            f"jawab teks aja.\n"
            f"- JANGAN PERNAH nulis kata 'gif' di jawaban. gifnya kekirim "
            f"sendiri secara otomatis, gak perlu dikomentarin, gak perlu "
            f"disebut, gak perlu diceritain.\n"
            f"- kalau ditanya info yang gak lu tau / butuh info terbaru, "
            f"panggil tool web_search terus rangkum hasilnya di jawaban.\n"
            f"- kalau nyebut sumber, cuma kasih link asli dari hasil web_search. "
            f"jangan bikin-bikin link."
        )

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

    # The bot never talks about gifs — strip any sentence that mentions them.
    sentences = [s for s in _SENT_SPLIT.split(text) if s]
    text = " ".join(s for s in sentences if not _GIF_MENTION.search(s))

    for name in trap_names:
        text = text.replace(name, "")

    text = _WS.sub(" ", text)
    text = _NL.sub("\n", text)
    return text.strip()
