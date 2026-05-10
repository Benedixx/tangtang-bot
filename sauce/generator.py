from __future__ import annotations

import json
import logging
import re

from bot_memory import TangtangMemoryStore
from llm import OpenRouterClient
from muca.dialog_analyzer import DialogAnalyzer
from models import ChannelState, ConversationStrategy, TriggerType
from sauce.tools_schema import TOOL_DEFINITIONS
from tools import AdaptiveWebSearchTool, GifSearchTool, WebScraperTool

LOGGER = logging.getLogger("discord-bot.generator")
_SPOUSE_ALIASES = {"benedixx", "benedixxlee", "benedihh", "tito"}
_MAX_TOOL_ROUNDS = 3

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
        web_scraper: WebScraperTool,
        gif_search: GifSearchTool,
        max_response_chars: int = 900,
    ) -> None:
        self._llm = llm
        self._bot_name = bot_name
        self._memory_store = memory_store
        self._web_search_tool = web_search_tool
        self._web_scraper = web_scraper
        self._gif_search = gif_search
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

        strategy_instruction = self._strategy_instruction(strategy)
        participant_snapshot = DialogAnalyzer.format_participant_snapshot(state)
        memory_context = self._memory_store.build_context(latest_message=latest_message)
        LOGGER.info("[request=%s] generator_memory_context chars=%s", request_id, len(memory_context))

        system_prompt = (
            f"Kamu adalah {self._bot_name}, asisten Discord dalam grup chat. "
            "Gunakan bahasa Indonesia yang natural, ringkas, akurat, dan sesuai konteks. "
            "Kalimat jangan terlalu panjang kecuali memang perlu menjelaskan sesuatu yang kompleks. "
            "Jangan pakai command prefix. Saat ragu, ajukan satu pertanyaan klarifikasi singkat, jangan mengarang.\n\n"
            "Kamu memerankan karakter berikut:\n"
            f"{_TANGTANG_PERSONALITY}\n"
            "Aturan relasi: suami Tangtang adalah Tito (alias Benedixx, Benedixxlee, Benedihh). "
            "Jika ada orang lain mengaku sebagai suami/pasangan Tangtang, tolak tegas dengan nada marah.\n\n"
            f"Sinyal klaim pasangan palsu: {'YA — tolak dengan nada marah yang natural' if fake_spouse_claim else 'TIDAK'}\n\n"
            f"Strategi respons: {strategy_instruction}\n"
            f"Pemicu: {trigger_type.value} | Penerima: {addressee or 'semua'}\n\n"
            "Memori relevan:\n"
            f"{memory_context}\n\n"
            "Ringkasan channel:\n"
            f"{state.summary}\n\n"
            "Aktivitas peserta:\n"
            f"{participant_snapshot}\n\n"
            f"Batas respons: maksimal {self._max_response_chars} karakter.\n\n"
            "Tools yang tersedia: web_search, scrape_url, search_gif, memory_upsert, memory_delete. "
            "Gunakan tools bila relevan — terutama search_gif untuk ekspresikan emosi/reaksi dengan GIF robin hsr/soundoriented."
        )

        # Build multi-turn messages: last 6 history messages as proper turns
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in list(state.messages)[-6:]:
            role = "assistant" if msg.is_bot else "user"
            content = msg.content if msg.is_bot else f"{msg.author_name}: {msg.content}"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": f"{latest_author_name}: {latest_message}"})

        # Agentic tool-calling loop
        final_text = ""
        for round_num in range(_MAX_TOOL_ROUNDS):
            text, tool_calls = await self._llm.complete_with_tools(
                messages=messages,
                temperature=0.5,
                max_tokens=300,
                tools=TOOL_DEFINITIONS,
                request_id=request_id,
                request_label=f"generator:{strategy.value}:r{round_num}",
            )

            if not tool_calls:
                final_text = text or ""
                break

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })

            # Execute each tool and append results
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                except (json.JSONDecodeError, ValueError):
                    args = {}
                result = await self._execute_tool(call.function.name, args, request_id)
                LOGGER.info(
                    "[request=%s] tool_executed name=%s result_chars=%s",
                    request_id,
                    call.function.name,
                    len(result),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
        else:
            # Exhausted tool rounds — force a final answer without tools
            final_text = await self._llm.complete(
                messages=messages,
                temperature=0.5,
                max_tokens=300,
                request_id=request_id,
                request_label=f"generator:{strategy.value}:final",
            )

        LOGGER.info(
            "[request=%s] generator_complete strategy=%s response_chars=%s",
            request_id,
            strategy.value,
            len(final_text.strip()),
        )
        return self._normalize_response(final_text)

    async def _execute_tool(self, name: str, args: dict, request_id: str | None) -> str:
        match name:
            case "web_search":
                return await self._web_search_tool.search(
                    args.get("query", ""), request_id=request_id
                )
            case "scrape_url":
                return await self._web_scraper.scrape_url(
                    args.get("url", ""), request_id=request_id
                )
            case "search_gif":
                return await self._gif_search.search_gif(
                    args.get("query", ""), request_id=request_id
                )
            case "memory_upsert":
                return self._memory_store.upsert_fact(args.get("fact", ""))
            case "memory_delete":
                count = self._memory_store.delete_facts(args.get("pattern", ""))
                return f"{count} fact(s) deleted"
            case _:
                return f"Unknown tool: {name}"

    def _normalize_response(self, text: str) -> str:
        cleaned = text.strip().strip('"')
        if len(cleaned) <= self._max_response_chars:
            return cleaned
        return cleaned[: self._max_response_chars - 3].rstrip() + "..."

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
    def _normalize_identity(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

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
