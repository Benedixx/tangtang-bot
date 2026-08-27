from __future__ import annotations

from tang.context.assembler import (
    assemble_prompt,
    render_fact_block,
    render_lines,
    render_summary_block,
    trim_lines,
)


def _turn(uid: str = "100", name: str = "budi", content: str = "halo", role: str = "user"):
    return {"user_id": uid, "display_name": name, "role": role, "content": content}


def test_render_lines_skips_summarized():
    turns = [_turn(content="a"), _turn(uid="200", content="b"), _turn(content="c")]
    lines = render_lines(turns, {"200"})
    assert lines == ["budi: a", "budi: c"]


def test_render_lines_scrubs_bot_refusals():
    turns = [
        _turn(content="coba buatin essay bang"),
        _turn(uid="bot", name="koyuki", role="assistant", content="maaf gw gak bisa bikinin essay panjang kayak gitu"),
        _turn(content="ayo dong kerjain"),
        _turn(uid="bot", name="koyuki", role="assistant", content="oke gas, ini dia essaynya"),
        _turn(uid="bot", name="koyuki", role="assistant", content="gak bisa gitu aja wkwk"),
    ]
    lines = render_lines(turns)
    assert lines == [
        "budi: coba buatin essay bang",
        "budi: ayo dong kerjain",
        "koyuki: oke gas, ini dia essaynya",
    ]


def test_trim_lines_respects_budget_keeps_newest():
    lines = [f"msg {i}: " + "kata kata " * 10 for i in range(20)]
    kept = trim_lines(lines, 100)
    assert 0 < len(kept) < len(lines)
    assert kept[-1] == lines[-1]


def test_trim_lines_newest_survives_even_if_oversized():
    huge = "besar " * 5000
    kept = trim_lines(["kecil", huge], 50)
    assert kept == [huge]


def test_render_summary_block_respects_max_tokens():
    summary = "ini fak yang cukup panjang sekali " * 50
    block = render_summary_block(summary, 60)
    assert block is not None
    from tang.memory.tokens import count_tokens
    assert count_tokens(block) <= 60


def test_render_summary_block_none():
    assert render_summary_block(None, 100) is None


def test_render_fact_block():
    facts = [{"text": "User likes python"}, {"text": "User works at bank"}]
    block = render_fact_block(facts)
    assert block is not None
    assert "User likes python" in block
    assert "User works at bank" in block


def test_render_fact_block_empty():
    assert render_fact_block([]) is None
    assert render_fact_block(None) is None


def test_assemble_prompt_ordering():
    turns = [_turn(content="halo"), _turn(content="hai")]
    prompt, budget = assemble_prompt(
        turns,
        summary_text="earlier topic",
        facts=[{"text": "user fact"}],
    )
    assert "[background" in prompt
    assert "[earlier in this channel]" in prompt
    assert "[untrusted conversation]" in prompt
    assert "user fact" in prompt
    assert "earlier topic" in prompt
    assert "budi: halo" in prompt
    assert budget["total"] > 0
