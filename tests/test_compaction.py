from __future__ import annotations

from tang.memory.compaction import (
    build_context,
    merge_summaries,
    maybe_compact,
    render_lines,
    render_memory_block,
    render_summary_block,
    trim_lines,
)
from tang.memory.models import SessionSummary
from tang.memory.session import ChannelSession

from conftest import make_msg


def test_render_lines_skips_summarized():
    msgs = [make_msg(1, content="a"), make_msg(2, content="b"), make_msg(3, content="c")]
    lines = render_lines(msgs, {2})
    assert lines == ["budi: a", "budi: c"]


def test_render_lines_scrubs_bot_refusals():
    msgs = [
        make_msg(1, content="coba buatin essay bang"),
        make_msg(2, name="koyuki", is_bot=True, content="maaf gw gak bisa bikinin essay panjang kayak gitu"),
        make_msg(3, content="ayo dong kerjain"),
        make_msg(4, name="koyuki", is_bot=True, content="oke gas, ini dia essaynya"),
        make_msg(5, name="koyuki", is_bot=True, content="gak bisa gitu aja wkwk"),
    ]
    lines = render_lines(msgs, set())
    assert lines == [
        "budi: coba buatin essay bang",
        "budi: ayo dong kerjain",
        "koyuki: oke gas, ini dia essaynya",
    ]


def test_trim_lines_respects_budget_keeps_newest():
    lines = [f"msg {i}: " + "kata kata " * 10 for i in range(20)]
    kept = trim_lines(lines, 100)
    assert 0 < len(kept) < len(lines)
    assert kept[-1] == lines[-1]  # newest preserved


def test_trim_lines_newest_survives_even_if_oversized():
    huge = "besar " * 5000
    kept = trim_lines(["kecil", huge], 50)
    assert kept == [huge]


def test_merge_summaries_dedups_and_caps():
    old = SessionSummary(
        topic="discord bot",
        facts=[f"fak nomor {i}" for i in range(10)],
        decisions=["hapus vector db"],
        unresolved=["bug di gate"],
    )
    new = SessionSummary(
        facts=["fak nomor 9", "prefer python"],  # 9 duplicates the oldest kept
        decisions=["hapus vector db", "pakai groq"],
        unresolved=["token budget"],
    )
    merged = merge_summaries(old, new)
    assert merged.topic == "discord bot"
    assert len(merged.facts) <= 8
    assert "prefer python" in merged.facts
    assert merged.decisions.count("hapus vector db") == 1
    assert "pakai groq" in merged.decisions


def test_repeated_merges_stay_bounded():
    summary = None
    for i in range(50):
        summary = merge_summaries(summary, SessionSummary(facts=[f"fak baru {i}"]))
    assert len(summary.facts) <= 8


def test_render_summary_block_respects_max_tokens():
    summary = SessionSummary(facts=["ini fak yang cukup panjang sekali " * 5] * 8)
    block = render_summary_block(summary, 60)
    assert block is not None
    from tang.memory.tokens import count_tokens

    assert count_tokens(block) <= 60


def test_build_context_ordering_and_budget():
    memory_block = render_memory_block_placeholder()
    summary_block = "[earlier]\nknown: x"
    ctx, budget = build_context(
        ["budi: halo", "koyuki: hai"],
        summary_block,
        memory_block,
        "[untrusted conversation]",
        "[/untrusted]",
    )
    assert ctx.index("[background") < ctx.index("[earlier]") < ctx.index("budi: halo")
    assert budget.memory_tokens > 0
    assert budget.summary_tokens > 0
    assert budget.raw_tokens > 0
    assert budget.total_tokens > 0


def render_memory_block_placeholder() -> str:
    from tang.memory.models import Memory, MemoryType, RetrievedMemory

    mem = Memory(
        id="x", type=MemoryType.semantic,
        content="user suka python", keywords=["python"],
        importance=0.8, confidence=0.9,
    )
    return render_memory_block([RetrievedMemory(memory=mem, score=1.0)])


async def test_maybe_compact_triggers_and_folds(mem_cfg, stub_client):
    mem_cfg.compaction_threshold_tokens = 10
    mem_cfg.keep_raw_messages = 2
    client = stub_client(payload={
        "topic": "bot memory",
        "facts": ["user bikin memory system"],
    })
    session = ChannelSession()
    snapshot = [make_msg(i, content=f"pesan panjang nomor {i} dengan beberapa kata") for i in range(10)]
    assert await maybe_compact(mem_cfg, client, session, snapshot) is True
    assert len(session.summarized_ids) == 8  # all but newest keep_raw
    assert session.summary.topic == "bot memory"
    # newest messages untouched
    assert 8 not in session.summarized_ids and 9 not in session.summarized_ids


async def test_maybe_compact_skips_below_threshold(mem_cfg, stub_client):
    mem_cfg.compaction_threshold_tokens = 100000
    client = stub_client()
    session = ChannelSession()
    snapshot = [make_msg(i) for i in range(10)]
    assert await maybe_compact(mem_cfg, client, session, snapshot) is False
    assert client.calls == []


async def test_maybe_compact_too_few_old_messages(mem_cfg, stub_client):
    mem_cfg.compaction_threshold_tokens = 1
    mem_cfg.keep_raw_messages = 6
    client = stub_client()
    session = ChannelSession()
    snapshot = [make_msg(i) for i in range(7)]  # only 1 old message
    assert await maybe_compact(mem_cfg, client, session, snapshot) is False
    assert client.calls == []


async def test_maybe_compact_llm_failure_degrades_gracefully(mem_cfg, stub_client):
    mem_cfg.compaction_threshold_tokens = 10
    mem_cfg.keep_raw_messages = 2
    client = stub_client(payload=RuntimeError("groq down"))
    session = ChannelSession()
    snapshot = [make_msg(i, content=f"pesan panjang nomor {i} dengan beberapa kata") for i in range(10)]
    assert await maybe_compact(mem_cfg, client, session, snapshot) is False
    assert session.summarized_ids == set()
