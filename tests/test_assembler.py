from __future__ import annotations

from tang.context.assembler import (
    assemble_prompt,
    render_fact_block,
    render_lines,
    trim_lines,
)


def _turn(uid: str = "100", name: str = "budi", content: str = "halo", role: str = "user"):
    return {"user_id": uid, "display_name": name, "role": role, "content": content}


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
    prompt, budget = assemble_prompt(turns, facts=[{"text": "user fact"}])
    assert prompt.index("[background") < prompt.index("[untrusted conversation]")
    assert "user fact" in prompt
    assert "budi: halo" in prompt
    assert budget["total"] > 0


def test_stale_turns_dropped_and_reply_to_marked():
    old = {**_turn(content="jam 4 ngangkat"), "timestamp": "2026-09-05T08:00:00+00:00"}
    new = {**_turn(content="kasih paham tang"), "timestamp": "2026-09-19T05:18:00+00:00"}
    prompt, _ = assemble_prompt([old, new], reply_to="budi: kasih paham tang")
    assert "jam 4 ngangkat" not in prompt
    assert prompt.endswith("bales pesan ini:\n[untrusted conversation]\nbudi: kasih paham tang\n[/untrusted]")
