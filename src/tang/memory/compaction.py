from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from pydantic import ValidationError
from rapidfuzz.fuzz import token_sort_ratio

from ..groq import GroqClient
from ..queue import Msg
from .models import RetrievedMemory, SessionSummary
from .tokens import count_tokens

LOGGER = logging.getLogger("tang.memory.compaction")

_MAX_LIST_ITEMS = 8

# The bot's own past refusals poison the context: seeing "maaf gw gak bisa"
# in history makes it refuse again. Never feed them back to the model.
_BOT_REFUSAL = re.compile(
    r"\b(gak|ga|nggak|ngga|tidak)\s+(bisa|mau|sanggup|kuat)\b", re.IGNORECASE
)


@dataclass(slots=True)
class ContextBudget:
    memory_tokens: int = 0
    summary_tokens: int = 0
    raw_tokens: int = 0
    total_tokens: int = 0


_COMPACT_SYSTEM = """\
You maintain a rolling summary of a casual Discord conversation so the bot \
can remember earlier context. You get older messages that are about to be \
dropped. Merge them into the existing summary if one is provided.
Keep only durable information: topic, established facts, decisions, open questions.
Drop small talk, greetings, jokes and repetition.
Output JSON only:
{"topic": "short phrase or null", "facts": ["..."], "decisions": ["..."], "unresolved": ["..."]}
Each list item must be one short line. Omit empty fields."""


def render_lines(msgs: list[Msg], summarized_ids: set[int]) -> list[str]:
    return [
        f"{m.author_name}: {m.content}"
        for m in msgs
        if m.message_id not in summarized_ids
        and not (m.is_bot and _BOT_REFUSAL.search(m.content))
    ]


def trim_lines(lines: list[str], budget_tokens: int) -> list[str]:
    """Keep the newest lines that fit the budget."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = count_tokens(line) + 1
        if kept and used + cost > budget_tokens:
            break
        if not kept:
            # Newest message always survives, even if oversized;
            # anything over budget stops older messages from joining.
            kept.append(line)
            if cost > budget_tokens:
                break
            used += cost
            continue
        kept.append(line)
        used += cost
    kept.reverse()
    return kept


def render_memory_block(memories: list[RetrievedMemory]) -> str | None:
    if not memories:
        return None
    bullets = "\n".join(f"- {r.memory.content}" for r in memories)
    return (
        f"[background notes about people here; may be stale, "
        f"use only if relevant]\n{bullets}"
    )


def render_summary_block(summary: SessionSummary | None, max_tokens: int) -> str | None:
    if summary is None or summary.is_empty():
        return None
    parts: list[str] = []
    if summary.topic:
        parts.append(f"topic: {summary.topic}")
    for label, items in (
        ("known", summary.facts),
        ("decided", summary.decisions),
        ("open", summary.unresolved),
    ):
        parts.extend(f"{label}: {item}" for item in items)
    block = "[earlier in this channel]\n" + "\n".join(parts)
    while parts and count_tokens(block) > max_tokens:
        parts.pop()
        block = "[earlier in this channel]\n" + "\n".join(parts)
    return block or None


def build_context(
    lines: list[str],
    summary_block: str | None,
    memory_block: str | None,
    history_open: str,
    history_close: str,
) -> tuple[str, ContextBudget]:
    sections: list[str] = []
    mem_tokens = sum_tokens = 0
    if memory_block:
        sections.append(memory_block)
        mem_tokens = count_tokens(memory_block)
    if summary_block:
        sections.append(summary_block)
        sum_tokens = count_tokens(summary_block)
    body = "\n".join(lines)
    sections.append(f"{history_open}\n{body}\n{history_close}")
    raw_tokens = count_tokens(body) + len(lines) + 2
    ctx = "\n\n".join(sections)
    return ctx, ContextBudget(
        memory_tokens=mem_tokens,
        summary_tokens=sum_tokens,
        raw_tokens=raw_tokens,
        total_tokens=count_tokens(ctx),
    )


def merge_summaries(old: SessionSummary | None, new: SessionSummary) -> SessionSummary:
    """Field-wise merge with bounded lists — prevents unbounded summary growth."""
    if old is None:
        return new
    merged = SessionSummary(topic=new.topic or old.topic)
    for src_a, src_b, dst in (
        (old.facts, new.facts, "facts"),
        (old.decisions, new.decisions, "decisions"),
        (old.unresolved, new.unresolved, "unresolved"),
    ):
        combined: list[str] = []
        for item in [*src_b, *src_a]:
            norm = " ".join(item.lower().split())
            if any(token_sort_ratio(" ".join(c.lower().split()), norm) >= 85 for c in combined):
                continue
            combined.append(item.strip())
        setattr(merged, dst, combined[-_MAX_LIST_ITEMS:])
    return merged


async def maybe_compact(
    cfg,
    client: GroqClient,
    session,
    snapshot: list[Msg],
) -> bool:
    """Summarize the oldest uncompacted segment into the session's rolling
    summary. Returns True if compaction happened."""
    lines = render_lines(snapshot, session.summarized_ids)
    raw_tokens = sum(count_tokens(line) + 1 for line in lines)
    if raw_tokens <= cfg.compaction_threshold_tokens:
        return False

    keep = cfg.keep_raw_messages
    old = [m for m in snapshot[:-keep] if m.message_id not in session.summarized_ids]
    if len(old) < 3:
        return False

    old_text = "\n".join(f"{m.author_name}: {m.content}" for m in old)
    existing = ""
    if session.summary is not None and not session.summary.is_empty():
        existing = session.summary.model_dump_json(exclude_none=True)

    messages = [
        {"role": "system", "content": _COMPACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"existing summary: {existing or '(none)'}\n\n"
                f"older messages to fold in:\n{old_text}"
            ),
        },
    ]
    try:
        data = await client.complete_json(
            messages, temperature=0.0, max_tokens=300, label="memory_compact",
        )
        parsed = SessionSummary.model_validate(data)
    except Exception:
        LOGGER.exception("compaction_failed channel_msgs=%s", len(old))
        return False

    session.summary = merge_summaries(session.summary, parsed)
    session.summarized_ids.update(m.message_id for m in old)
    LOGGER.info("compaction_done folded=%s", len(old))
    return True
