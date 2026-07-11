from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord

from collections import deque

from ..bubble import bubbles
from ..engage import should_engage
from ..models import ChannelState, ChatMessage, TriggerType

if TYPE_CHECKING:
    from bot import TangBot

LOGGER = logging.getLogger("tang.chat")

_HARD_TRIGGERS = {
    TriggerType.DIRECT_MENTION,
    TriggerType.REPLY_TO_BOT,
    TriggerType.NAME_MENTION,
    TriggerType.BOT_PREFIX,
}

_MAX_TOOL_ROUNDS = 3


class ChatHandler:
    """Handles the core chat logic: dispatch, trigger detection, generation."""

    def __init__(self, bot: TangBot) -> None:
        self.bot = bot
        self.config = bot.config
        self.llm = bot.llm
        self.groq = bot.groq
        self.memory = bot.memory
        self.guardrail = bot.guardrail
        self.tool_schemas, self.tool_executors = bot.tools

        self._prefix = self.config.bot_prefix  # "!k"
        self._max_context = self.config.max_context_messages
        self._max_chars = self.config.max_response_chars
        self._cooldown = self.config.response_cooldown_seconds
        self._channels: dict[int, ChannelState] = {}
        self._state_lock = asyncio.Lock()
        self._name_patterns: list[re.Pattern[str]] = []
        self._system_prompt = self._load_prompt()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_prompt(self) -> str:
        path = Path(__file__).parent.parent / "prompts" / "tang.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        LOGGER.warning("prompts/tang.md not found, using default")
        return "You are Tang, a Discord AI assistant."

    def build_name_patterns(self, names: list[str]) -> None:
        seen: set[str] = set()
        for name in names:
            cleaned = name.strip().lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            self._name_patterns.append(re.compile(rf"(^|\W){re.escape(name.strip())}(\W|$)", re.IGNORECASE))

    # ------------------------------------------------------------------
    # Channel state
    # ------------------------------------------------------------------

    def _get_state(self, channel_id: int) -> ChannelState:
        state = self._channels.get(channel_id)
        if state is None:
            state = ChannelState(
                messages=deque(maxlen=self._max_context)
            )
            self._channels[channel_id] = state
        return state

    async def _add_message(self, channel_id: int, msg: ChatMessage) -> None:
        async with self._state_lock:
            state = self._get_state(channel_id)
            state.messages.append(msg)
            state.total_count += 1
            if not msg.is_bot:
                state.participants[msg.author_id] = msg.author_name

    async def _add_bot_message(self, channel_id: int, msg: ChatMessage) -> None:
        async with self._state_lock:
            state = self._get_state(channel_id)
            state.messages.append(msg)
            state.total_count += 1
            state.bot_message_ids.add(msg.message_id)
            state.last_response_at = msg.created_at

    async def _snapshot(self, channel_id: int) -> ChannelState:
        async with self._state_lock:
            state = self._get_state(channel_id)
            return ChannelState(
                messages=deque(state.messages, maxlen=self._max_context),
                total_count=state.total_count,
                summary=state.summary,
                summary_trigger=state.summary_trigger,
                bot_message_ids=set(state.bot_message_ids),
                last_response_at=state.last_response_at,
                participants=dict(state.participants),
            )

    # ------------------------------------------------------------------
    # Trigger detection
    # ------------------------------------------------------------------

    def detect_trigger(self, message: discord.Message, state: ChannelState) -> TriggerType:
        content = message.content or ""

        if self.bot.user and self.bot.user in message.mentions:
            return TriggerType.DIRECT_MENTION

        if (message.author.bot or message.webhook_id is not None) and content.startswith(self._prefix):
            return TriggerType.BOT_PREFIX

        if self._is_reply_to_bot(message, state):
            return TriggerType.REPLY_TO_BOT

        if self._is_name_mention(content):
            return TriggerType.NAME_MENTION

        return TriggerType.CONTEXTUAL

    def _is_reply_to_bot(self, message: discord.Message, state: ChannelState) -> bool:
        ref = message.reference
        if ref is None:
            return False
        # Check via state tracking
        if ref.message_id is not None and ref.message_id in state.bot_message_ids:
            return True
        # Check via resolved reference
        resolved = ref.resolved
        if isinstance(resolved, discord.Message) and self.bot.user is not None:
            return resolved.author.id == self.bot.user.id
        return False

    def _is_name_mention(self, content: str) -> bool:
        if not self._name_patterns:
            return False
        return any(p.search(content) for p in self._name_patterns)

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if self.bot.user is None or message.author.id == self.bot.user.id:
            return

        # Other bots/webhooks: only respond via !k prefix
        if message.author.bot or message.webhook_id is not None:
            if not (message.content or "").startswith(self._prefix):
                return

        channel_id = message.channel.id
        content = (message.content or "").strip()
        request_id = f"msg-{message.id}"

        # Strip !k prefix from bot/webhook messages
        if (message.author.bot or message.webhook_id is not None) and content.startswith(self._prefix):
            content = content[len(self._prefix):].strip()

        if not content:
            return

        # Add message to state
        await self._add_message(channel_id, ChatMessage(
            message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            content=content,
            created_at=message.created_at,
            is_bot=message.author.bot or message.webhook_id is not None,
        ))

        # Snapshot state for async generation
        state = await self._snapshot(channel_id)

        # Trigger detection
        trigger = self.detect_trigger(message, state)
        LOGGER.info("[%s] trigger=%s channel=%s author=%s", request_id, trigger.value, channel_id, message.author.display_name)

        # Guardrail check
        is_attack, rejection = await self.guardrail.check(content, request_id)
        if is_attack:
            LOGGER.warning("[%s] guardrail_blocked author=%s", request_id, message.author.display_name)
            if trigger in _HARD_TRIGGERS and rejection:
                try:
                    await message.reply(rejection, mention_author=False)
                except discord.HTTPException:
                    pass
            return

        # Engagement decision
        engage, reason = await should_engage(
            trigger=trigger,
            state=state,
            message=content,
            groq=self.groq,
            cooldown_seconds=self._cooldown,
            request_id=request_id,
        )
        LOGGER.info("[%s] engage=%s reason=%s", request_id, engage, reason)

        if not engage:
            return

        # Enqueue generation — processed sequentially by queue worker
        addressee = (
            message.author.display_name
            if trigger in _HARD_TRIGGERS
            else None
        )
        LOGGER.info("[%s] queue_enqueue channel=%s", request_id, channel_id)
        await self.bot.gen_queue.enqueue(self._generate_and_send(
            message=message,
            trigger=trigger,
            state=state,
            content=content,
            author_name=message.author.display_name,
            addressee=addressee,
            request_id=request_id,
        ))

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def _generate_and_send(
        self,
        message: discord.Message,
        trigger: TriggerType,
        state: ChannelState,
        content: str,
        author_name: str,
        addressee: str | None,
        request_id: str,
    ) -> None:
        started = time.perf_counter()
        channel_id = message.channel.id

        try:
            async with message.channel.typing():
                # Build context
                system = self._build_system_prompt(state, content, author_name, addressee)
                memory_ctx = self.memory.build_context(message=content)
                if memory_ctx:
                    system += f"\n\nRelevant memories:\n{memory_ctx}"

                # Build conversation messages
                msgs = [{"role": "system", "content": system}]
                for m in list(state.messages)[-8:]:
                    role = "assistant" if m.is_bot else "user"
                    if m.is_bot:
                        msgs.append({"role": role, "content": m.content})
                    else:
                        msgs.append({"role": role, "content": f"{m.author_name}: {m.content}"})
                msgs.append({"role": "user", "content": f"{author_name}: {content}"})

                # Tool loop (Groq for fast tool decisions)
                if self.groq is not None:
                    msgs = await self._tool_loop(msgs, request_id)

                # Stream final response from OpenRouter
                stream = self.llm.stream_complete(
                    messages=msgs,
                    temperature=0.5,
                    max_tokens=500,
                    request_id=request_id,
                    request_label="generate",
                )

                # Split into bubbles and send
                # Only prefix with !k when responding to a bot/webhook (inter-bot protocol)
                prefix = self._prefix if (message.author.bot or message.webhook_id is not None) else ""
                sent_messages: list[discord.Message] = []
                async for bubble_text in bubbles(stream, bot_prefix=prefix):
                    try:
                        if trigger in _HARD_TRIGGERS and not sent_messages:
                            sent = await message.reply(bubble_text, mention_author=False)
                        else:
                            sent = await message.channel.send(bubble_text)
                        sent_messages.append(sent)
                    except discord.HTTPException:
                        LOGGER.exception("[%s] send_failed", request_id)

                if not sent_messages:
                    LOGGER.info("[%s] empty_response duration_ms=%s", request_id,
                                int((time.perf_counter() - started) * 1000))
                    return

                # Register bot messages in state
                for sent in sent_messages:
                    await self._add_bot_message(channel_id, ChatMessage(
                        message_id=sent.id,
                        author_id=self.bot.user.id,
                        author_name=self.bot.user.display_name,
                        content=sent.content,
                        created_at=sent.created_at,
                        is_bot=True,
                    ))

                LOGGER.info(
                    "[%s] done bubbles=%s duration_ms=%s",
                    request_id, len(sent_messages),
                    int((time.perf_counter() - started) * 1000),
                )

        except Exception:
            LOGGER.exception("[%s] generation_failed", request_id)
            if trigger in _HARD_TRIGGERS:
                try:
                    await message.reply(
                        "Sorry, had a hiccup. Try again?",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass

    async def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        request_id: str,
    ) -> list[dict[str, Any]]:
        """Run tool-calling loop using Groq (up to 3 rounds)."""
        if self.groq is None:
            return messages

        for rnd in range(_MAX_TOOL_ROUNDS):
            _, tool_calls = await self.groq.complete_with_tools(
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                tools=self.tool_schemas,
                request_id=request_id,
                request_label=f"tools:r{rnd}",
            )

            if not tool_calls:
                break

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = __import__("json").loads(tc.function.arguments)
                except Exception:
                    args = {}

                executor = self.tool_executors.get(name)
                if executor is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = await executor(request_id=request_id, **args)
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                        LOGGER.exception("[%s] tool_error name=%s", request_id, name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:1500] if len(result) > 1500 else result,
                })

        return messages

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        state: ChannelState,
        latest_message: str,
        author_name: str,
        addressee: str | None,
    ) -> str:
        # Base system prompt
        prompt = self._system_prompt

        # Channel context
        participant_info = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                state.participants.items(),
                key=lambda x: sum(1 for m in state.messages if not m.is_bot and m.author_id == x[0]),
                reverse=True,
            )[:5]
        ) if state.participants else "No other participants yet."

        prompt_parts = [
            prompt,
            "",
            "---",
            f"Channel participants: {participant_info}",
        ]

        if state.summary:
            prompt_parts.append(f"Channel summary: {state.summary[:300]}")
        if addressee:
            prompt_parts.append(f"Message directed at you from: {addressee}")

        prompt_parts.append(
            "Respond naturally. You can send multiple short messages for impact — "
            "each message should be 1-3 sentences. Use GIFs sparingly, only when the moment really calls for it."
        )

        return "\n".join(prompt_parts)
