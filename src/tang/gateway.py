from __future__ import annotations

import asyncio
import logging
import re
import time

import discord
from rapidfuzz.fuzz import token_sort_ratio

from .config import Config
from .context.long_term import LongTermMemory
from .context.short_term import ShortTermBuffer
from .context.summarizer import Summarizer
from .filters import tier0_reason
from .gate import Gate
from .groq import GroqClient
from .guard import Guard
from .memory.fact_extractor import FactExtractor
from .memory.triggers import wants_memory
from .persona import PersonaBuilder, sanitize
from .queue import ChannelRegistry, Job, Msg, Reply
from .responder import Responder
from .storage.json_store import JsonStore
from .tools.gifs import GifStore
from .tools.search import WebSearchTool
from .trap import TrapGuard

LOGGER = logging.getLogger("tang.gateway")

# Idle timer for extraction tracking per channel
_extraction_timers: dict[int, asyncio.TimerHandle] = {}
_extraction_counters: dict[int, int] = {}


class Gateway(discord.Client):
    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config

        # Storage
        self.json_store = JsonStore(config.memory.data_dir)

        # Memory layers
        self.buffer = ShortTermBuffer(self.json_store)
        self.summarizer = Summarizer(self.json_store, GroqClient(config.groq_api_key, config.models.gate))
        self.long_term = LongTermMemory(self.json_store, config.memory)
        self.fact_extractor = FactExtractor(GroqClient(config.groq_api_key, config.models.gate))

        # Core components
        self.trap = TrapGuard(config.trap)
        self.registry = ChannelRegistry(
            self,
            base_threshold=config.chat.base_threshold,
            debounce_s=config.chat.debounce_s,
        )
        self.gate_groq = GroqClient(config.groq_api_key, config.models.gate)
        self.gate = Gate(self.gate_groq, config.chat)
        self.guard = Guard()
        self.persona = PersonaBuilder(config)
        self.gif_store = GifStore(config.gif.dir, config.gif.manifest)
        self.search = WebSearchTool()
        self.responder = Responder(
            config.groq_api_key,
            config.models.responder,
            self.persona,
            self.gif_store,
            self.search,
            self.json_store,
            config.memory,
        )

        self._allowed = frozenset(config.chat.allowed_channels)
        names = [config.bot_name, *config.bot_name_aliases]
        self._name_patterns = [
            re.compile(rf"(^|\W){re.escape(n)}(\W|$)", re.IGNORECASE)
            for n in names
            if n.strip()
        ]
        self._trap_names: frozenset[str] = frozenset()
        self._forget_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        LOGGER.info(
            "ready user=%s id=%s",
            self.user, self.user.id if self.user else None,
        )
        self._trap_names = self._collect_trap_names()
        if self._forget_task is None or self._forget_task.done():
            if self.config.memory.enabled:
                self.long_term.sweep()
                self._forget_task = asyncio.get_running_loop().create_task(
                    self._forget_loop()
                )
        await self.gif_store.upload_all(self, self.config.gif.staging_channel)

    async def _forget_loop(self) -> None:
        while True:
            await asyncio.sleep(86400.0)
            try:
                self.long_term.sweep()
            except Exception:
                LOGGER.exception("fact_sweep_failed")

    async def close(self) -> None:
        """Flush pending memory extraction before shutting down."""
        try:
            pending = [
                cid for cid, count in _extraction_counters.items()
                if count > 0
            ]
            if pending:
                await asyncio.gather(*(self._idle_extract(cid) for cid in pending))
                LOGGER.info("shutdown_memory_flushed channels=%s", len(pending))
        except Exception:
            LOGGER.exception("shutdown_memory_flush_failed")
        finally:
            await super().close()

    def _collect_trap_names(self) -> frozenset[str]:
        ids = set(self.config.trap.channels)
        names: set[str] = set()
        for guild in self.guilds:
            for ch in guild.channels:
                if ch.id in ids and isinstance(ch, discord.TextChannel):
                    names.add(ch.name.lower())
        return frozenset(names)

    # ------------------------------------------------------------------
    # Message pipeline
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if self.user is None or message.author.id == self.user.id:
            return

        # Trap guard — terminal.
        if message.guild is not None and self.trap.is_trap(message.channel.id):
            await self.trap.handle(message, self.user.id)
            return

        if not self._tracked(message):
            return

        msg = Msg(
            message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            content=(message.content or "").strip(),
            is_bot=message.author.bot or message.webhook_id is not None,
            guild_id=message.guild.id if message.guild else None,
        )
        if not msg.content:
            return

        channel_id = message.channel.id
        request_id = f"msg-{message.id}"
        state = self.registry.append(channel_id, msg)
        self._note_engagement(message, state)

        # Persist to short-term buffer
        if self.config.memory.enabled:
            self.buffer.append(
                channel_id=channel_id,
                guild_id=msg.guild_id,
                user_id=str(msg.author_id),
                display_name=msg.author_name,
                role="assistant" if msg.is_bot else "user",
                content=msg.content,
            )

        # Explicit "remember this" request — extract now, don't wait for idle.
        if (
            self.config.memory.enabled
            and not msg.is_bot
            and wants_memory(msg.content)
        ):
            LOGGER.info("[%s] memory_request_detected channel=%s", request_id, channel_id)
            _extraction_counters[channel_id] = self.config.memory.fact_extraction_min_messages
            asyncio.get_running_loop().create_task(self._idle_extract(channel_id))

        # Injection scan — canned reply, terminal.
        if self.guard.scan(msg.content):
            LOGGER.warning(
                "[%s] injection_attempt channel=%s author=%s content=%s",
                request_id, channel_id, message.author.id, msg.content[:120],
            )
            if self._is_forced(message):
                await self._send_canned(message)
            return

        forced = self._is_forced(message)

        # Tier 0 filter — drop the trigger, keep the buffer.
        reason = tier0_reason(message, self.config.chat)
        if reason:
            LOGGER.info("[%s] tier0_drop channel=%s reason=%s", request_id, channel_id, reason)
            return

        self.registry.schedule(channel_id, forced, message.id)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user is not None and payload.user_id == self.user.id:
            return
        state = self.registry.get(payload.channel_id)
        if state.watch_until <= time.time():
            return
        if any(m.message_id == payload.message_id and m.is_bot for m in state.buffer):
            state.watch_engaged = True

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    async def process(self, job) -> Reply | None:
        state = self.registry.get(job.channel_id)
        request_id = f"job-{job.trigger_message_id}"

        if not await self.gate.decide(job, state, time.time(), request_id):
            return None

        memories: list[dict] = []
        if self.config.memory.enabled:
            # Trigger summarization if buffer is large
            turns = self.buffer.read(job.channel_id)
            if turns:
                await self.summarizer.maybe_compact(
                    job.channel_id,
                    turns,
                    token_threshold=self.config.memory.buffer_max_tokens,
                    keep_recent=5,
                )

            # Search long-term memory for relevant facts
            recent_human = [
                t.get("content", "") for t in reversed(turns)
                if t.get("role") != "assistant"
            ][:3]
            if recent_human:
                query = " ".join(reversed(recent_human))
                # Search for facts from the trigger author
                user_facts = self.long_term.search(
                    str(job.trigger_author_id), query
                )
                memories = [f["fact"] for f in user_facts]

        result = await self.responder.run(
            job, state, request_id,
            channel_id=job.channel_id,
            memories=memories,
        )
        if result is None:
            return None

        reply_text = sanitize(result.text, self._trap_names)
        gif = (result.gif or "").strip()
        if gif and gif in reply_text:
            reply_text = reply_text.replace(gif, "").strip()
        if not reply_text and not gif:
            return None

        # Anti-repeat guard: never send what the bot already said recently.
        recent_bot = [m.content for m in reversed(state.buffer) if m.is_bot][:5]
        if reply_text and any(
            token_sort_ratio(reply_text.lower(), prev.lower()) >= 80
            for prev in recent_bot if prev
        ):
            LOGGER.info("[%s] repeat_reply_dropped", request_id)
            return None

        return Reply(text=reply_text, gif=gif)

    async def deliver(self, job, reply: Reply) -> None:
        channel = self.get_channel(job.channel_id)
        if channel is None or self.user is None:
            return
        state = self.registry.get(job.channel_id)

        content = reply.text
        if reply.gif:
            content = f"{reply.text}\n{reply.gif}" if reply.text else reply.gif

        try:
            if reply.anchored:
                ref = channel.get_partial_message(job.trigger_message_id)
                sent = await channel.send(content, reference=ref)
            else:
                sent = await channel.send(content)
        except discord.HTTPException:
            LOGGER.exception("send_failed channel=%s", job.channel_id)
            return

        state.last_reply_ts = time.time()
        if reply.gif and self.gif_store.contains(reply.gif):
            state.recent_gifs.append(reply.gif)

        # Persist bot reply to buffer
        if self.config.memory.enabled:
            self.buffer.append(
                channel_id=job.channel_id,
                guild_id=job.snapshot[-1].guild_id if job.snapshot else None,
                user_id=str(self.user.id),
                display_name=self.user.display_name,
                role="assistant",
                content=reply.text,
            )

        self.registry.append(job.channel_id, Msg(
            message_id=sent.id,
            author_id=self.user.id,
            author_name=self.user.display_name,
            content=reply.text,
            is_bot=True,
        ))

        if not job.forced:
            state.interjections.append(time.time())
            self.gate.start_watch(state, time.time())

        if self.config.memory.enabled:
            _extraction_counters[job.channel_id] = _extraction_counters.get(job.channel_id, 0) + 1
            self._arm_extraction(job.channel_id)

    # ------------------------------------------------------------------
    # Long-term memory extraction
    # ------------------------------------------------------------------

    def _arm_extraction(self, channel_id: int) -> None:
        """Arm or re-arm the idle extraction timer."""
        if channel_id in _extraction_timers:
            _extraction_timers[channel_id].cancel()

        idle_s = self.config.memory.fact_extraction_idle_minutes * 60
        loop = asyncio.get_running_loop()
        _extraction_timers[channel_id] = loop.call_later(
            idle_s, lambda: asyncio.get_running_loop().create_task(
                self._idle_extract(channel_id)
            )
        )

    async def _idle_extract(self, channel_id: int) -> None:
        try:
            count = _extraction_counters.get(channel_id, 0)
            min_msgs = self.config.memory.fact_extraction_min_messages
            if count < min_msgs:
                return
            _extraction_counters[channel_id] = 0

            if channel_id in _extraction_timers:
                _extraction_timers[channel_id].cancel()
                del _extraction_timers[channel_id]

            turns = self.buffer.read(channel_id)[-12:]
            if not turns:
                return

            LOGGER.info("fact_extract_started channel=%s turns=%s", channel_id, len(turns))

            # Determine user_id and guild_id from turns
            human_turns = [t for t in turns if t.get("role") != "assistant"]
            if not human_turns:
                return

            last_human = human_turns[-1]
            user_id = last_human.get("user_id", "")
            display_name = last_human.get("display_name", "user")
            guild_id = None
            if turns:
                guild_id_str = turns[-1].get("guild_id")
                if guild_id_str:
                    try:
                        guild_id = int(guild_id_str)
                    except (ValueError, TypeError):
                        pass

            candidates = await self.fact_extractor.extract(
                turns, channel_id, guild_id
            )
            if not candidates:
                LOGGER.info("fact_extract channel=%s extracted=0", channel_id)
                return

            stored, updated, rejected = self.long_term.remember(
                user_id, display_name, candidates, guild_id
            )
            LOGGER.info(
                "fact_extract channel=%s extracted=%s stored=%s updated=%s rejected=%s",
                channel_id, len(candidates), stored, updated, rejected,
            )
        except Exception:
            LOGGER.exception("idle_extract_failed channel=%s", channel_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tracked(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return self.config.chat.dm_allowed
        if message.guild is None:
            return False
        return not self._allowed or message.channel.id in self._allowed

    def _is_forced(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return True
        if self.user is not None and self.user in message.mentions:
            return True
        ref = message.reference
        if ref is not None:
            if ref.message_id is not None:
                state = self.registry.get(message.channel.id)
                if any(m.message_id == ref.message_id and m.is_bot for m in state.buffer):
                    return True
            resolved = ref.resolved
            if isinstance(resolved, discord.Message) and self.user is not None:
                if resolved.author.id == self.user.id:
                    return True
        content = message.content or ""
        return any(p.search(content) for p in self._name_patterns)

    def _note_engagement(self, message: discord.Message, state) -> None:
        if state.watch_until <= time.time():
            return
        ref = message.reference
        if ref is None or ref.message_id is None:
            return
        if any(m.message_id == ref.message_id and m.is_bot for m in state.buffer):
            state.watch_engaged = True

    async def _send_canned(self, message: discord.Message) -> None:
        try:
            await message.reply(self.guard.canned(), mention_author=False)
        except discord.HTTPException:
            pass
