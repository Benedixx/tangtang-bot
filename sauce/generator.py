from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator

from bot_memory import TangtangMemoryStore
from llm import GroqClient, OpenRouterClient
from models import ChannelState, ConversationStrategy, TriggerType
from muca.dialog_analyzer import DialogAnalyzer
from sauce.tools_schema import TOOL_DEFINITIONS
from tools import AdaptiveWebSearchTool, GifSearchTool, WebScraperTool

LOGGER = logging.getLogger("discord-bot.generator")
_SPOUSE_ALIASES = {"benedixx", "benedixxlee", "benedihh", "tito"}
_MAX_TOOL_ROUNDS = 3
_CHUNK_SPLIT_RE = re.compile(r"([.!?]\s+|\n+)")

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
        groq: GroqClient | None = None,
    ) -> None:
        self._llm = llm
        self._groq = groq
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
        fake_spouse_claim = self._is_fake_spouse_claim(
            latest_author_name, latest_message
        )
        if fake_spouse_claim:
            LOGGER.info("[request=%s] generator_signal=fake_spouse_claim", request_id)

        strategy_instruction = self._strategy_instruction(strategy)
        participant_snapshot = DialogAnalyzer.format_participant_snapshot(state)
        memory_context = self._memory_store.build_context(latest_message=latest_message)
        LOGGER.info(
            "[request=%s] generator_memory_context chars=%s",
            request_id,
            len(memory_context),
        )

        system_prompt = (
            f"Kamu adalah {self._bot_name}, anggota grup chat Discord. "
            "Bukan asisten, bukan chatbot — kamu ya kamu, Tangtang, ngobrol seperti biasa. "
            "Gunakan bahasa Indonesia yang natural, ringkas, dan sesuai konteks. "
            "Jangan pakai command prefix. Saat ragu, tanya singkat, jangan mengarang.\n\n"
            "Karakter:\n"
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
            "Tools: web_search, scrape_url, search_gif, memory_upsert, memory_delete. Gunakan bila relevan.\n"
            "GIF — sesekali boleh pakai search_gif kalau momen beneran pas (reaksi kuat, situasi absurd/lucu/dramatis). "
            "Tapi jangan dipaksakan. Kalau kirim GIF, langsung drop URL-nya di baris baru tanpa kata pengantar — "
            "kayak orang kirim GIF di chat, bukan presentasi. Discord auto-embed URL .gif."
        )

        # Build multi-turn messages: last 6 history messages as proper turns
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in list(state.messages)[-6:]:
            role = "assistant" if msg.is_bot else "user"
            content = msg.content if msg.is_bot else f"{msg.author_name}: {msg.content}"
            messages.append({"role": role, "content": content})
        messages.append(
            {"role": "user", "content": f"{latest_author_name}: {latest_message}"}
        )

        # Tool-calling loop: Groq drives decisions (fast), OpenRouter writes final answer (quality)
        tool_client = self._groq if self._groq else self._llm
        for round_num in range(_MAX_TOOL_ROUNDS):
            _, tool_calls = await tool_client.complete_with_tools(
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                tools=TOOL_DEFINITIONS,
                request_id=request_id,
                request_label=f"generator:tools:r{round_num}",
            )

            if not tool_calls:
                break

            # Append assistant message with tool calls
            messages.append(
                {
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
                }
            )

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        # Final response always from OpenRouter regardless of whether tools were used
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

    async def generate_response_chunks(
        self,
        *,
        strategy: ConversationStrategy,
        trigger_type: TriggerType,
        state: ChannelState,
        latest_author_name: str,
        latest_message: str,
        addressee: str | None,
        request_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        fake_spouse_claim = self._is_fake_spouse_claim(
            latest_author_name, latest_message
        )
        if fake_spouse_claim:
            LOGGER.info("[request=%s] generator_signal=fake_spouse_claim", request_id)

        strategy_instruction = self._strategy_instruction(strategy)
        participant_snapshot = DialogAnalyzer.format_participant_snapshot(state)
        memory_context = self._memory_store.build_context(latest_message=latest_message)
        LOGGER.info(
            "[request=%s] generator_memory_context chars=%s",
            request_id,
            len(memory_context),
        )

        system_prompt = (
            f"Kamu adalah {self._bot_name}, anggota grup chat Discord. "
            "Bukan asisten, bukan chatbot \u2014 kamu ya kamu, Tangtang, ngobrol seperti biasa. "
            "Gunakan bahasa Indonesia yang natural, ringkas, dan sesuai konteks. "
            "Jangan pakai command prefix. Saat ragu, tanya singkat, jangan mengarang.\n\n"
            "Karakter:\n"
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
            "Panjang respons: natural, 1-3 kalimat per ide. Tiap kalimat padat. "
            "Kalau ada list, max 3-4 poin singkat saja.\n\n"
            "Saat pakai web_search atau scrape_url: jelaskan hasilnya dengan kata-katamu sendiri \u2014 "
            "gaya santai Tangtang, bukan copy-paste raw data. "
            "Sertakan sumber di akhir dengan format hyperlink: [nama singkat](url).\n\n"
            "Tools: web_search, scrape_url, search_gif, memory_upsert, memory_delete.\n"
            "GIF \u2014 sesekali boleh kirim GIF kalau momen beneran pas: reaksi kuat, situasi lucu/absurd/dramatis. "
            "Jangan dipaksakan setiap ada emosi. Kalau kirim GIF, langsung drop URL-nya di baris baru "
            "tanpa kata pengantar \u2014 kayak orang kirim GIF di chat, bukan presentasi."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in list(state.messages)[-6:]:
            role = "assistant" if msg.is_bot else "user"
            content = msg.content if msg.is_bot else f"{msg.author_name}: {msg.content}"
            messages.append({"role": role, "content": content})
        messages.append(
            {"role": "user", "content": f"{latest_author_name}: {latest_message}"}
        )

        tool_client = self._groq if self._groq else self._llm
        for round_num in range(_MAX_TOOL_ROUNDS):
            _, tool_calls = await tool_client.complete_with_tools(
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                tools=TOOL_DEFINITIONS,
                request_id=request_id,
                request_label=f"generator:tools:r{round_num}",
            )

            if not tool_calls:
                break

            messages.append(
                {
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
                }
            )

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        buffer = ""
        chunk_count = 0
        async for delta in self._llm.stream_complete(
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            request_id=request_id,
            request_label=f"generator:{strategy.value}:stream",
        ):
            buffer += delta
            parts = _CHUNK_SPLIT_RE.split(buffer)
            n_complete = (len(parts) - 1) // 2
            for i in range(n_complete):
                raw = (parts[2 * i] + parts[2 * i + 1]).strip()
                chunk = self._normalize_chunk(raw)
                if chunk and len(chunk) >= 3:
                    chunk_count += 1
                    yield chunk
            buffer = parts[-1]

        if buffer.strip():
            chunk = self._normalize_chunk(buffer.strip())
            if chunk:
                chunk_count += 1
                yield chunk

        # Some reasoning models stream thinking tokens internally but emit zero delta.content
        # (all tokens go to native_tokens_reasoning). Fall back to non-streaming complete().
        if chunk_count == 0:
            LOGGER.info(
                "[request=%s] stream_empty_fallback=true strategy=%s",
                request_id,
                strategy.value,
            )
            final_text = await self._llm.complete(
                messages=messages,
                temperature=0.5,
                max_tokens=500,
                request_id=request_id,
                request_label=f"generator:{strategy.value}:fallback",
            )
            fallback = self._normalize_response(final_text)
            if fallback:
                fallback_parts = _CHUNK_SPLIT_RE.split(fallback)
                n_fb = (len(fallback_parts) - 1) // 2
                for i in range(n_fb):
                    raw = (fallback_parts[2 * i] + fallback_parts[2 * i + 1]).strip()
                    chunk = self._normalize_chunk(raw)
                    if chunk and len(chunk) >= 3:
                        chunk_count += 1
                        yield chunk
                tail = self._normalize_chunk(fallback_parts[-1].strip())
                if tail:
                    chunk_count += 1
                    yield tail

        LOGGER.info(
            "[request=%s] generator_stream_complete strategy=%s chunks=%s",
            request_id,
            strategy.value,
            chunk_count,
        )

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
        cleaned = re.sub(r"!\[.*?\]\((https?://\S+)\)", r"\1", cleaned)
        cleaned = re.sub(r"\*?\*?\[GIF\]\*?\*?\s*", "", cleaned).strip()
        if len(cleaned) <= self._max_response_chars:
            return cleaned
        return cleaned[: self._max_response_chars - 3].rstrip() + "..."

    @staticmethod
    def _extract_gif_url(result: str) -> str | None:
        m = re.search(r"URL:\s*(https?://\S+\.gif)", result)
        return m.group(1) if m else None

    @staticmethod
    def _normalize_chunk(text: str) -> str:
        cleaned = text.strip().strip('"')
        cleaned = re.sub(r"!\[.*?\]\((https?://\S+)\)", r"\1", cleaned)
        cleaned = re.sub(r"\*?\*?\[GIF\]\*?\*?\s*", "", cleaned).strip()
        return cleaned

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

        if re.search(
            r"\b(aku|saya|gw|gue|gua|i am|i'm|im)\b.*\b(suami|husband|pasangan|pacar)\b",
            compact,
        ):
            return True

        return bool(
            re.search(r"\b(aku|saya|i am|i'm|im)\s+(benedixx|benedihh|tito)\b", compact)
        )

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
            return "Buat ringkasan diskusi saat ini yang singkat namun berguna, lalu beri satu saran langkah lanjut yang logis."
        if strategy == ConversationStrategy.IN_CONTEXT_CHIME_IN:
            return "Ikut menimpali hanya jika memberi nilai tambah: klarifikasi, fakta berguna, atau fasilitasi diskusi secara ringkas."
        return "Tetap singkat dan netral."
