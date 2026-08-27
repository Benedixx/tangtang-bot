from __future__ import annotations

import re
from typing import Any

from ..memory.tokens import count_tokens

_HISTORY_OPEN = "[untrusted conversation]"
_HISTORY_CLOSE = "[/untrusted]"

_BOT_REFUSAL = re.compile(
    r"\b(gak|ga|nggak|ngga|tidak)\s+(bisa|mau|sanggup|kuat)\b", re.IGNORECASE
)


def render_lines(
    turns: list[dict[str, Any]],
    summarized_ids: set[str] | None = None,
) -> list[str]:
    """Render turns as 'name: content' lines, filtering out summarized and refusal turns."""
    lines = []
    for t in turns:
        uid = t.get("user_id", "")
        if summarized_ids and uid in summarized_ids:
            continue
        content = t.get("content", "")
        if t.get("role") == "assistant" and _BOT_REFUSAL.search(content):
            continue
        name = t.get("display_name", "user")
        lines.append(f"{name}: {content}")
    return lines


def trim_lines(lines: list[str], budget_tokens: int) -> list[str]:
    """Keep the newest lines that fit the budget."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = count_tokens(line) + 1
        if kept and used + cost > budget_tokens:
            break
        kept.append(line)
        if not kept or cost <= budget_tokens:
            used += cost
        if cost > budget_tokens:
            break
    kept.reverse()
    return kept


def render_summary_block(summary_text: str | None, max_tokens: int) -> str | None:
    if not summary_text:
        return None
    block = f"[earlier in this channel]\n{summary_text}"
    while count_tokens(block) > max_tokens and len(block) > 50:
        block = block[: block.rfind("\n")]
    return block or None


def render_fact_block(facts: list[dict[str, Any]] | None) -> str | None:
    if not facts:
        return None
    bullets = "\n".join(f"- {f.get('text', '')}" for f in facts)
    return (
        f"[background notes about people here; may be stale, "
        f"use only if relevant]\n{bullets}"
    )


def assemble_prompt(
    turns: list[dict[str, Any]],
    summary_text: str | None = None,
    facts: list[dict[str, Any]] | None = None,
    recent_max_tokens: int = 4000,
    summary_max_tokens: int = 250,
    summarized_ids: set[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Assemble the full prompt from buffer, summary, and facts.

    Returns (prompt_text, budget_dict).
    """
    lines = render_lines(turns, summarized_ids)
    lines = trim_lines(lines, recent_max_tokens)

    sections: list[str] = []
    budget: dict[str, int] = {}

    fact_block = render_fact_block(facts)
    if fact_block:
        sections.append(fact_block)
        budget["facts"] = count_tokens(fact_block)

    summary_block = render_summary_block(summary_text, summary_max_tokens)
    if summary_block:
        sections.append(summary_block)
        budget["summary"] = count_tokens(summary_block)

    body = "\n".join(lines)
    sections.append(f"{_HISTORY_OPEN}\n{body}\n{_HISTORY_CLOSE}")
    budget["raw"] = count_tokens(body) + len(lines) + 2

    prompt = "\n\n".join(sections)
    budget["total"] = count_tokens(prompt)

    return prompt, budget
