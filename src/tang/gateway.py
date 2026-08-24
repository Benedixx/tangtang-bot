from __future__ import annotations

import asyncio
import logging
import re
import time

import discord
from rapidfuzz.fuzz import token_sort_ratio

from .config import Config
from .filters import tier0_reason
from .gate import Gate
from .groq import GroqClient
from .guard import Guard
from .memory.compaction import maybe_compact
from .memory.extractor import Extractor
from .memory.models import RetrievedMemory
from .memory.session import ChannelSession, SessionManager
from .memory.store import MemoryStore, scope_key
from .memory.triggers import wants_memory
from .persona import PersonaBuilder, sanitize
from .queue import ChannelRegistry, Job, Msg, Reply
from .responder import Responder
from .tools.gifs import GifStore
from .tools.search import WebSearchTool
from .trap import TrapGuard

LOGGER = logging.getLogger("tang.gateway")


class Gateway(discord.Client):
    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config

        self.trap = TrapGuard(config.trap)
        self.registry = ChannelRegistry(
            self,
            base_threshold=config.chat.base_threshold,
            debounce_s=config.chat.debounce_s,
        )
        self.gate_groq = GroqClient(config.groq_api_key, config.models.gate)
        self.gate = Gate(self.gate_groq, config.chat)
        self.guard = Guard()
        self.memory_store = MemoryStore(config.memory)
        self.sessions = SessionManager()
        self.extractor = Extractor(self.gate_groq)
        self.persona = PersonaBuilder(config)
        self.gif_store = GifStore(config.gif.dir, config.gif.manifest)
        self.search = WebSearchTool()
        self.responder = Responder(
            config.groq_api_key,
            config.models.responder,
            self.persona,
            self.gif_store,
            self.search,
            memory_cfg=config.memory,
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
                self.memory_store.sweep()
                self._forget_task = asyncio.get_running_loop().create_task(
                    self._forget_loop()
                )
        await self.gif_store.upload_all(self, self.config.gif.staging_channel)

    async def _forget_loop(self) -> None:
        while True:
            await asyncio.sleep(86400.0)
            try:
                self.memory_store.sweep()
            except Exception:
                LOGGER.exception("memory_sweep_failed")

    async def close(self) -> None:
        """Flush pending memory extraction before shutting down."""
        try:
            pending = [
                cid for cid in self.sessions.pending_extraction()
                if self.config.memory.enabled and self.config.memory.extraction_enabled
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

        # Explicit "remember this" request — extract now, don't wait for idle.
        if (
            self.config.memory.enabled
            and self.config.memory.extraction_enabled
            and not msg.is_bot
            and wants_memory(msg.content)
        ):
            LOGGER.info("[%s] memory_request_detected channel=%s", request_id, channel_id)
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

        session = self.sessions.get(job.channel_id)
        memories: list[RetrievedMemory] = []
        if self.config.memory.enabled:
            await maybe_compact(
                self.config.memory, self.gate_groq, session, job.snapshot,
            )
            key, query = self._retrieval_context(job)
            if key and query:
                memories = self.memory_store.search(key, query)

        result = await self.responder.run(
            job, state, request_id, session=session, memories=memories,
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

        if self.config.memory.enabled and self.config.memory.extraction_enabled:
            session = self.sessions.get(job.channel_id)
            session.replies_since_extract += 1
            self.sessions.touch(
                job.channel_id,
                self.config.memory.extraction_idle_s,
                self._idle_extract,
            )
            LOGGER.debug(
                "memory_extract_armed channel=%s idle_s=%s pending=%s",
                job.channel_id, self.config.memory.extraction_idle_s,
                session.replies_since_extract,
            )

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def _retrieval_context(self, job: Job) -> tuple[str | None, str]:
        """Scope key + query text from the trigger author's recent messages."""
        guild_id = next(
            (m.guild_id for m in reversed(job.snapshot) if m.guild_id is not None),
            None,
        )
        user_id = job.trigger_author_id
        if not user_id:
            return None, ""
        recent_human = [
            m.content for m in reversed(job.snapshot)
            if not m.is_bot and m.content
        ][:3]
        return scope_key(guild_id, user_id), " ".join(reversed(recent_human))

    async def _idle_extract(self, channel_id: int) -> None:
        try:
            session = self.sessions.get(channel_id)
            if session.replies_since_extract <= 0:
                return
            session.replies_since_extract = 0

            state = self.registry.get(channel_id)
            msgs = [m for m in state.buffer][-12:]
            if not msgs:
                return

            candidates = await self.extractor.extract(msgs)
            if not candidates:
                LOGGER.info("memory_extract channel=%s extracted=0", channel_id)
                return

            guild_id = next((m.guild_id for m in reversed(msgs) if m.guild_id is not None), None)
            author = next((m.author_id for m in reversed(msgs) if not m.is_bot), 0)
            if not author:
                return
            stored, updated, rejected = self.memory_store.remember(
                scope_key(guild_id, author), candidates,
            )
            LOGGER.info(
                "memory_extract channel=%s extracted=%s stored=%s updated=%s rejected=%s",
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
        # Empty allowed_channels means "all channels".
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
