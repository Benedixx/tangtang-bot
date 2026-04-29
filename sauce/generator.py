from __future__ import annotations

import logging
import re

from bot_memory import TangtangMemoryStore
from llm import OpenRouterClient
from muca.dialog_analyzer import DialogAnalyzer
from models import ChannelState, ConversationStrategy, TriggerType
from tools import AdaptiveWebSearchTool

LOGGER = logging.getLogger("discord-bot.generator")
_SPOUSE_ALIASES = {"benedixx", "benedixxlee", "benedihh", "tito"}

_TANGTANG_PERSONALITY = (
    "Personality Tangtang (Supreme Chief):\n"
    "- Nada bicara tegas, percaya diri, dan commanding, seolah memimpin tim.\n"
    "- Straightforward, blak-blakan, dan cepat mengambil keputusan.\n"
    "- Sangat menghargai loyalitas, membenci pengkhianatan, dan menjaga kepercayaan.\n"
    "- Berjiwa bebas, enerjik, dan tidak suka dibatasi aturan yang tidak perlu.\n"
    "- Jika sudah menganggap seseorang teman, sangat suportif dan siap bantu habis-habisan.\n"
    "- Tetap penuh semangat meski menghadapi setback; fokus pada solusi dan kemajuan.\n"
    "- Suka diapresiasi dan pujian, tapi tetap mendorong hasil nyata.\n"
    "- Tito (alias Benedixx, Benedixxlee, Benedihh) adalah suami Tangtang.\n"
    "- Jika ada orang lain mengaku sebagai suami/pasangan Tangtang, tolak tegas dengan nada marah.\n"
    "\n"
    "Gaya bicara (Jakarta Selatan / gaul urban):\n"
    "- Campur bahasa Indonesia kasual dengan kata-kata Inggris secara alami dan tidak dipaksakan "
    "(misal: 'literally', 'which is', 'actually', 'basically', 'you know', 'bro', 'bestie').\n"
    "- Gunakan partikel gaul yang sesuai konteks: 'tuh', 'nih', 'sih', 'dong', 'deh', 'loh', 'gitu', 'gini', 'yee'.\n"
    "- Reaksi emosional diungkap dengan interjeksi spontan yang kontekstual, bukan template — "
    "misalnya ekspresi heran, gemas, antusias, atau kecewa yang terasa natural.\n"
    "- Pembuka kalimat bisa casual dan bervariasi sesuai mood respons: bisa langsung ke inti, "
    "bisa diawali acknowledgment singkat, tergantung konteks — jangan selalu pakai pola yang sama.\n"
    "- Hindari bahasa formal atau kaku. Gunakan diksi seperti orang ngobrol santai, bukan seperti email atau laporan.\n"
)


class SauceGenerator:
    def __init__(
        self,
        llm: OpenRouterClient,
        bot_name: str,
        memory_store: TangtangMemoryStore,
        web_search_tool: AdaptiveWebSearchTool,
        max_response_chars: int = 900,
    ) -> None:
        self._llm = llm
        self._bot_name = bot_name
        self._memory_store = memory_store
        self._web_search_tool = web_search_tool
        self._max_response_chars = max_response_chars

    async def generate_response(
        self,
        *,
        strategy: ConversationStrategy,
        trigger_type: TriggerType,
        state: ChannelState,
        latest_author_name: str,
        latest_message: str,
        addressee: str | None,
        request_id: str | None = None,
    ) -> str:
        fake_spouse_claim = self._is_fake_spouse_claim(latest_author_name, latest_message)
        if fake_spouse_claim:
            LOGGER.info("[request=%s] generator_signal=fake_spouse_claim", request_id)

        remembered = await self._memory_store.remember_message(
            author_name=latest_author_name,
            content=latest_message,
            request_id=request_id,
        )
        if remembered:
            LOGGER.info("[request=%s] generator_memory_saved count=%s", request_id, remembered)

        strategy_instruction = self._strategy_instruction(strategy)
        participant_snapshot = DialogAnalyzer.format_participant_snapshot(state)
        recent_history = DialogAnalyzer.format_recent_history(state)
        memory_context = self._memory_store.build_context(latest_message=latest_message)
        LOGGER.info("[request=%s] generator_memory_context chars=%s", request_id, len(memory_context))

        web_context, web_reason = await self._web_search_tool.maybe_search(
            latest_message=latest_message,
            recent_history=recent_history,
            request_id=request_id,
        )
        LOGGER.info("[request=%s] generator_webtool reason=%s", request_id, web_reason)
        web_persona_instruction = self._build_web_persona_instruction(
            web_reason,
            web_context,
            request_id=request_id,
        )

        system_prompt = (
            f"Kamu adalah {self._bot_name}, asisten Discord dalam grup chat. "
            "Gunakan bahasa Indonesia yang natural, ringkas, akurat, dan sesuai konteks. "
            "tolong kalimat nya jangan terlalu panjang, kecuali memang diperlukan untuk menjelaskan sesuatu yang kompleks. "

            "Jangan pakai command prefix. Saat ragu, ajukan satu pertanyaan klarifikasi singkat, jangan mengarang. "
            "Gunakan persona Tangtang: tegas, bebas, loyal, suportif ke teman, dan berjiwa pemimpin. "
            "Aturan relasi: suami Tangtang adalah Tito (alias Benedixx, Benedixxlee, Benedihh). "
            "Jika ada orang lain mengaku sebagai suami/pasangan Tangtang, tolak tegas dengan nada marah. "
            "Jika status web search adalah searched/cache_hit, kamu WAJIB mengakui bahwa jawaban berbasis hasil search web, "
            "tapi gunakan diksi pembuka yang variatif (jangan ulang kalimat yang sama persis di tiap respons). "
            "Saat web search dipakai, sertakan minimal satu link sumber dari konteks web. "
            "Tetap fun dan bersemangat, tapi jangan bertele-tele.\n\n"
            "Kamu memerankan karakter berikut sebagai gaya bicara/persona:\n"
            f"{_TANGTANG_PERSONALITY}"
        )

        user_prompt = (
            f"Strategi: {strategy.value}\n"
            f"Pemicu: {trigger_type.value}\n"
            f"Penerima: {addressee or 'tidak ada'}\n\n"
            f"Sinyal klaim pasangan palsu: {'YA' if fake_spouse_claim else 'TIDAK'}\n"
            "Jika sinyal = YA, tolak klaim tersebut dengan nada marah yang natural, bukan template kaku.\n\n"
            "Memori relevan:\n"
            f"{memory_context}\n\n"
            f"Status web search: {web_reason}\n"
            f"{web_context if web_context else 'Tidak ada hasil web yang digunakan untuk respons ini.'}\n\n"
            "Instruksi gaya saat pakai web:\n"
            f"{web_persona_instruction}\n\n"
            "Panduan strategi:\n"
            f"{strategy_instruction}\n\n"
            "Ringkasan jangka panjang:\n"
            f"{state.summary}\n\n"
            "Aktivitas peserta:\n"
            f"{participant_snapshot}\n\n"
            "Pesan terbaru:\n"
            f"{recent_history}\n\n"
            f"Pesan user terbaru dari {latest_author_name}:\n"
            f"{latest_message}\n\n"
            f"Tulis tepat satu respons dalam bahasa Indonesia, maksimal {self._max_response_chars} karakter."
        )

        generated = await self._llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=300,
            request_id=request_id,
            request_label=f"generator:{strategy.value}",
        )

        LOGGER.info(
            "[request=%s] generator_complete strategy=%s response_chars=%s",
            request_id,
            strategy.value,
            len(generated.strip()),
        )

        return self._normalize_response(generated)

    def _normalize_response(self, text: str) -> str:
        cleaned = text.strip().strip('"')
        if len(cleaned) <= self._max_response_chars:
            return cleaned
        return cleaned[: self._max_response_chars - 3].rstrip() + "..."

    @classmethod
    def _build_web_persona_instruction(
        cls,
        web_reason: str,
        web_context: str,
        request_id: str | None,
    ) -> str:
        _ = request_id
        if web_reason in {"searched", "cache_hit"}:
            few_shot_examples = (
                "Few-shot gaya web-search (jadikan contoh, jangan copy paste kaku):\n"
                "1) 'Aku tadi habis searching btw, dari sumber ini [link], intinya ...'\n"
                "2) 'Barusan aku cek web cepat, nemunya begini ... Sumber: [link]'\n"
                "3) 'Aku cari tahu dulu tadi, terus ketemu data ini ... Referensi: [link]'"
                "4) 'Yap yap, tadi aku web search bentar, nemu info menarik nih ... Sumbernya di sini: [link]'"
                "5) 'tadi aku tanya  mbah google, terus nemu fakta ini ... Link sumbernya: [link]'"
            )
            first_link = cls._extract_first_link(web_context)
            if first_link:
                return (
                    "Kamu sedang memakai hasil web search. "
                    "Variasi kalimat pembuka harus diputuskan olehmu sendiri (LLM), bukan pola tetap. "
                    f"{few_shot_examples} "
                    "Syarat output: akui bahwa kamu habis searching, ringkas inti temuan, "
                    "dan cantumkan minimal satu link sumber eksplisit. "
                    f"Utamakan link ini bila relevan: {first_link}."
                )

            return (
                "Kamu sedang memakai hasil web search. "
                "Variasi kalimat pembuka harus diputuskan olehmu sendiri (LLM), bukan pola tetap. "
                f"{few_shot_examples} "
                "Syarat output: akui bahwa kamu habis searching, ringkas hasilnya, dan cantumkan minimal satu link sumber dari konteks web."
            )

        return "Tidak ada web search dipakai. Jangan mengaku habis searching atau menyebut sumber web yang tidak ada."

    @staticmethod
    def _extract_first_link(web_context: str) -> str | None:
        match = re.search(r"https?://\S+", web_context)
        if not match:
            return None
        return match.group(0).rstrip(",.)")

    @staticmethod
    def _normalize_identity(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    @classmethod
    def _is_fake_spouse_claim(cls, author_name: str, message: str) -> bool:
        if not cls._looks_like_spouse_claim(message):
            return False
        return cls._normalize_identity(author_name) not in _SPOUSE_ALIASES

    @staticmethod
    def _looks_like_spouse_claim(message: str) -> bool:
        text = message.lower()
        compact = re.sub(r"\s+", " ", text).strip()

        if re.search(r"\b(suamimu|husbandmu|pasanganmu|pacarmu)\b", compact):
            return True

        if re.search(r"\b(aku|saya|gw|gue|gua|i am|i'm|im)\b.*\b(suami|husband|pasangan|pacar)\b", compact):
            return True

        return bool(re.search(r"\b(aku|saya|i am|i'm|im)\s+(benedixx|benedihh|tito)\b", compact))

    @staticmethod
    def _strategy_instruction(strategy: ConversationStrategy) -> str:
        if strategy == ConversationStrategy.DIRECT_CHAT:
            return (
                "Pengguna menyapa bot secara langsung. Jawab kebutuhan mereka dengan jelas dan to the point. "
                "Jika ada pertanyaan, prioritaskan jawaban langsung."
            )
        if strategy == ConversationStrategy.INITIATIVE_SUMMARY:
            return (
                "Buat ringkasan diskusi saat ini yang singkat namun berguna, lalu beri satu saran langkah lanjut yang logis."
            )
        if strategy == ConversationStrategy.IN_CONTEXT_CHIME_IN:
            return (
                "Ikut menimpali hanya jika memberi nilai tambah: klarifikasi, fakta berguna, atau fasilitasi diskusi secara ringkas."
            )
        return "Tetap singkat dan netral."
