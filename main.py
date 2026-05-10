from __future__ import annotations

import asyncio
import logging
import re
import time

import discord

from bot_memory import TangtangMemoryStore
from config import AppConfig, load_config
from llm import GroqClient, OpenRouterClient
from muca import DialogAnalyzer, StrategyArbitrator
from models import ChatMessage, ConversationStrategy, TriggerType
from sauce import SauceGenerator, SauceScheduler
from tools import AdaptiveWebSearchTool, GifSearchTool, WebScraperTool

LOGGER = logging.getLogger("discord-bot")
_MAX_LOG_PREVIEW_CHARS = 180

_HARD_TRIGGERS = {
    TriggerType.DIRECT_MENTION,
    TriggerType.REPLY_TO_BOT,
    TriggerType.NAME_MENTION,
}


class MucaSauceDiscordBot(discord.Client):
    def __init__(
        self,
        *,
        config: AppConfig,
        analyzer: DialogAnalyzer,
        arbitrator: StrategyArbitrator,
        scheduler: SauceScheduler,
        generator: SauceGenerator,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._analyzer = analyzer
        self._arbitrator = arbitrator
        self._scheduler = scheduler
        self._generator = generator
        self._channel_locks: dict[int, asyncio.Lock] = {}

        name_candidates = [self._config.bot_name, *self._config.bot_name_aliases]
        normalized_seen: set[str] = set()
        self._bot_name_patterns: list[re.Pattern[str]] = []

        for candidate in name_candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in normalized_seen:
                continue
            normalized_seen.add(lowered)

            escaped = re.escape(cleaned)
            self._bot_name_patterns.append(re.compile(rf"(^|\W){escaped}(\W|$)", re.IGNORECASE))

    async def on_ready(self) -> None:
        if self.user is not None:
            LOGGER.info("Connected as %s (%s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if self.user is None:
            return

        if message.author.bot or message.author.id == self.user.id:
            return

        content = (message.content or "").strip()
        if not content:
            return

        channel_id = message.channel.id
        request_id = f"msg-{message.id}"
        started = time.perf_counter()

        LOGGER.info(
            "[request=%s] incoming guild=%s channel=%s author=%s(%s) content=%s",
            request_id,
            message.guild.id if message.guild else "dm",
            channel_id,
            message.author.display_name,
            message.author.id,
            self._preview_text(content),
        )

        lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())

        async with lock:
            state = self._analyzer.get_state(channel_id)
            trigger_type = await self._detect_trigger_type(message, state, content)

            LOGGER.info("[request=%s] trigger=%s", request_id, trigger_type.value)

            self._analyzer.add_message(
                channel_id,
                ChatMessage(
                    message_id=message.id,
                    author_id=message.author.id,
                    author_name=message.author.display_name,
                    content=content,
                    created_at=message.created_at,
                    is_bot=False,
                ),
            )

            LOGGER.info(
                "[request=%s] context total_messages=%s window_messages=%s since_summary=%s",
                request_id,
                state.total_message_count,
                len(state.messages),
                state.message_count_since_summary,
            )

            strategy = self._arbitrator.select_strategy(
                trigger_type=trigger_type,
                state=state,
                latest_message=content,
            )
            recent_history = self._analyzer.format_recent_history(state, limit=self._config.max_context_messages)

            LOGGER.info("[request=%s] strategy=%s", request_id, strategy.value)

            should_engage, reason = await self._scheduler.should_engage(
                trigger_type=trigger_type,
                strategy=strategy,
                state=state,
                latest_message=content,
                recent_history=recent_history,
                request_id=request_id,
            )

            LOGGER.info(
                "[request=%s] scheduler should_engage=%s reason=%s",
                request_id,
                should_engage,
                reason,
            )

            if not should_engage:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                LOGGER.info("[request=%s] action=skip duration_ms=%s", request_id, elapsed_ms)
                return

            addressee = message.author.display_name if trigger_type in _HARD_TRIGGERS else None

            try:
                response = await self._generator.generate_response(
                    strategy=strategy,
                    trigger_type=trigger_type,
                    state=state,
                    latest_author_name=message.author.display_name,
                    latest_message=content,
                    addressee=addressee,
                    request_id=request_id,
                )
            except Exception:
                LOGGER.exception("[request=%s] generator_failed", request_id)
                if trigger_type in _HARD_TRIGGERS:
                    await message.reply(
                        "I had a temporary issue while generating a response. Please try again.",
                        mention_author=False,
                    )
                return

            if not response:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                LOGGER.info("[request=%s] action=empty_response duration_ms=%s", request_id, elapsed_ms)
                return

            LOGGER.info(
                "[request=%s] generated response_chars=%s preview=%s",
                request_id,
                len(response),
                self._preview_text(response),
            )

            sent = await self._send_response(message, response, trigger_type)
            if sent is None:
                LOGGER.info("[request=%s] action=send_failed", request_id)
                return

            self._analyzer.add_message(
                channel_id,
                ChatMessage(
                    message_id=sent.id,
                    author_id=self.user.id,
                    author_name=self.user.display_name,
                    content=response,
                    created_at=sent.created_at,
                    is_bot=True,
                ),
            )
            self._analyzer.mark_bot_message(channel_id, sent.id)

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            LOGGER.info(
                "[request=%s] action=respond sent_message_id=%s duration_ms=%s",
                request_id,
                sent.id,
                elapsed_ms,
            )

            if strategy == ConversationStrategy.INITIATIVE_SUMMARY:
                self._analyzer.set_summary(channel_id, response)
                LOGGER.info("[request=%s] summary_updated=true", request_id)

    async def _send_response(
        self,
        message: discord.Message,
        response: str,
        trigger_type: TriggerType,
    ) -> discord.Message | None:
        try:
            async with message.channel.typing():
                if trigger_type in _HARD_TRIGGERS:
                    return await message.reply(response, mention_author=False)
                return await message.channel.send(response)
        except discord.HTTPException:
            LOGGER.exception("Failed to send response")
            return None

    @staticmethod
    def _preview_text(content: str) -> str:
        normalized = content.replace("\n", " ").strip()
        if len(normalized) <= _MAX_LOG_PREVIEW_CHARS:
            return normalized
        return normalized[: _MAX_LOG_PREVIEW_CHARS - 3].rstrip() + "..."

    async def _detect_trigger_type(
        self,
        message: discord.Message,
        state,
        content: str,
    ) -> TriggerType:
        if self.user in message.mentions:
            return TriggerType.DIRECT_MENTION

        if await self._is_reply_to_bot(message, state):
            return TriggerType.REPLY_TO_BOT

        if self._is_name_mention(content):
            return TriggerType.NAME_MENTION

        return TriggerType.CONTEXTUAL

    async def _is_reply_to_bot(self, message: discord.Message, state) -> bool:
        if message.reference is None:
            return False

        referenced_id = message.reference.message_id
        if referenced_id is not None and referenced_id in state.bot_message_ids:
            return True

        resolved = message.reference.resolved
        if isinstance(resolved, discord.Message):
            return self.user is not None and resolved.author.id == self.user.id

        if referenced_id is None or not hasattr(message.channel, "fetch_message"):
            return False

        try:
            referenced = await message.channel.fetch_message(referenced_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return False

        return self.user is not None and referenced.author.id == self.user.id

    def _is_name_mention(self, content: str) -> bool:
        if not self._bot_name_patterns:
            return False
        return any(pattern.search(content) is not None for pattern in self._bot_name_patterns)


def _build_bot(config: AppConfig) -> MucaSauceDiscordBot:
    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True

    llm_client = OpenRouterClient(
        api_keys=config.openrouter_api_keys,
        model_name=config.model_name,
        site_name=config.openrouter_site_name,
        site_url=config.openrouter_site_url,
    )

    memory_store = TangtangMemoryStore(file_path=config.memory_file_path, llm=llm_client)
    web_search_tool = AdaptiveWebSearchTool(
        enabled=config.web_search_enabled,
        max_results=config.web_search_max_results,
        min_interval_seconds=config.web_search_min_interval_seconds,
    )
    web_scraper = WebScraperTool()
    gif_search = GifSearchTool(api_key=config.klipy_api_key)

    groq_client = GroqClient(api_key=config.groq_api_key, model=config.groq_model) if config.groq_api_key else None

    analyzer = DialogAnalyzer(max_context_messages=config.max_context_messages)
    arbitrator = StrategyArbitrator(summary_interval_messages=config.summary_interval_messages)
    scheduler = SauceScheduler(llm=llm_client, cooldown_seconds=config.response_cooldown_seconds, groq=groq_client)
    generator = SauceGenerator(
        llm=llm_client,
        bot_name=config.bot_name,
        memory_store=memory_store,
        web_search_tool=web_search_tool,
        web_scraper=web_scraper,
        gif_search=gif_search,
        max_response_chars=config.max_response_chars,
    )

    return MucaSauceDiscordBot(
        config=config,
        analyzer=analyzer,
        arbitrator=arbitrator,
        scheduler=scheduler,
        generator=generator,
        intents=intents,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_config()
    bot = _build_bot(config)
    bot.run(config.discord_token)


if __name__ == "__main__":
    main()
