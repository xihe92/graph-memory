from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from graph_memory import buffer
from graph_memory.anchors import CodeAnchor
from graph_memory.client import call
from graph_memory.config import load_config
from graph_memory.graduation import (
    Exchange,
    SegmentInput,
    build_extract_prompt,
    chunks,
    graduate,
    segment_input,
)
from graph_memory.knowledge import (
    Entity,
    EntityKind,
    ItemType,
    KnowledgeStore,
    Relation,
    fuzzy_safe,
)

from .conftest import topic_embed, turn
from .test_daemon import _start, _wait

T0 = datetime(2026, 11, 1, 10, tzinfo=UTC)


def seg(
    *pairs: tuple[str, str], trigger: str = "session_end", start: datetime = T0
) -> SegmentInput:
    xs = tuple(
        Exchange(f"x{start.day}{i}", start + timedelta(minutes=i), p, r)
        for i, (p, r) in enumerate(pairs)
    )
    return SegmentInput("s1", xs, trigger, "repo:/w/limiter", "limiter")


class Script:
    """A fake model: answers extraction and merge prompts from queues, records prompts."""

    def __init__(self, extracts: list[dict], merges: list[dict | Exception] | None = None) -> None:  # type: ignore[type-arg]
        self.extracts = list(extracts)
        self.merges = list(merges or [])
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "distil" in prompt:
            return json.dumps(self.extracts.pop(0))
        out = self.merges.pop(0) if self.merges else {"items": []}
        if isinstance(out, Exception):
            raise out
        return "```json\n" + json.dumps(out) + "\n```"


LOCK = {
    "entities": [
        {"name": "svc/limiter.py", "kind": "file", "summary": "rate limiter"},
        {"name": "Redis", "kind": "tool", "summary": "the shared store"},
    ],
    "items": [
        {
            "type": "decision",
            "statement": "svc/limiter.py guards acquire() with a Redis SETNX lock with 30s expiry",
            "why": "workers raced on the counter",
            "entities": ["svc/limiter.py", "Redis"],
            "scope": "project",
            "evidence": ["X1"],
        },
        {
            "type": "procedure",
            "statement": "acquire() retries 5 times with exponential backoff",
            "entities": ["svc/limiter.py"],
            "scope": "project",
            "evidence": ["X2"],
        },
        {"type": "nonsense", "statement": "dropped because the type is unknown"},
        {"type": "fact", "statement": "too short"},
    ],
}

LUA = {
    "entities": [
        {"name": "redis", "kind": "tool", "summary": ""},
        {"name": "svc/limiter.py", "kind": "file", "summary": ""},
    ],
    "items": [
        {
            "type": "decision",
            "statement": "svc/limiter.py uses an atomic Redis Lua script (INCR + EXPIRE) "
            "instead of a SETNX lock",
            "why": "the SETNX lock deadlocked under load",
            "entities": ["svc/limiter.py", "redis"],
            "scope": "project",
            "evidence": ["X1"],
        },
        {
            "type": "procedure",
            "statement": "acquire() retries 5 times with exponential backoff",
            "entities": ["svc/limiter.py"],
            "evidence": ["X1"],
        },
    ],
}


def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "k", topic_embed)


def test_graduation_builds_entities_items_and_edges(tmp_path: Path) -> None:
    k = store(tmp_path)
    ask = Script([LOCK])
    s = graduate(k, seg(("add a lock", "done"), ("add retries", "done")), ask, topic_embed)

    assert len(ask.prompts) == 2  # one extraction; one merge for the two new items
    assert {e.name for e in k.entities.values()} == {"svc/limiter.py", "Redis"}
    items = sorted(k.items.values(), key=lambda i: i.statement)
    assert [i.type for i in items] == [ItemType.PROCEDURE, ItemType.DECISION]
    lock = items[1]
    assert lock.scope == "repo:/w/limiter" and lock.why == "workers raced on the counter"
    assert lock.valid_at == T0 and [s.until for s in lock.sources] == [T0]
    assert s.items == [i.item_id for i in k.items.values()] and s.calls == 2
    # learned together and both about svc/limiter.py: a co-occurrence edge, no model needed
    (edge,) = k.edges.values()
    assert edge.relation is Relation.CO_OCCURS and edge.origin == "cooccur"
    assert s.exchanges == 2 and (s.start, s.end) == (T0, T0 + timedelta(minutes=1))


def test_second_session_reinforces_reuses_entities_and_invalidates(tmp_path: Path) -> None:
    k = store(tmp_path)
    graduate(k, seg(("add a lock", "done"), ("add retries", "done")), Script([LOCK]), topic_embed)
    old_lock = next(i for i in k.items.values() if "SETNX" in i.statement)
    retries = next(i for i in k.items.values() if "retries" in i.statement)

    later = T0 + timedelta(days=1)
    ask = Script(
        [LUA],
        [
            {
                "items": [
                    {
                        "id": "N1",
                        "duplicate_of": None,
                        "invalidates": [{"id": "K1", "reason": "the lock was replaced"}],
                        "related": [{"id": "K2", "relation": "depends_on", "strength": "weak"}],
                    }
                ]
            }
        ],
    )
    graduate(k, seg(("lock deadlocks, fix it", "replaced with lua"), start=later), ask, topic_embed)

    # exact and case-insensitive entity names are reused, not duplicated
    assert len(k.entities) == 2
    redis = next(e for e in k.entities.values() if e.name == "Redis")
    assert redis.mentions == 2
    # the identical statement was reinforced without asking the model about it
    assert retries.strength == 2.0 and len(retries.segments) == 2
    merge_prompt = ask.prompts[1]
    assert "\nN2 " not in merge_prompt and "SETNX" in merge_prompt
    # the contradicted item keeps its history
    new_lock = next(i for i in k.items.values() if "Lua" in i.statement)
    assert not old_lock.valid and old_lock.invalid_at == later
    assert old_lock.invalidated_by == new_lock.item_id
    sup = k.edges[(new_lock.item_id, old_lock.item_id, "supersedes")]
    assert sup.reason == "the lock was replaced"
    rel = [e for e in k.edges.values() if e.relation is Relation.DEPENDS_ON]
    assert len(rel) == 1 and rel[0].weight == 0.3
    assert k.stats()["items"] == 2 and k.stats()["invalidated"] == 1

    # everything survives a reload
    k.save()
    again = KnowledgeStore(tmp_path / "k", topic_embed)
    assert again.stats() == k.stats()
    assert again.entity_by_name("REDIS") is not None


def test_failed_merge_keeps_the_extracted_items(tmp_path: Path) -> None:
    k = store(tmp_path)
    down = RuntimeError("down")
    s = graduate(k, seg(("a", "b"), ("c", "d")), Script([LOCK], [down, down]), topic_embed)
    assert len(s.items) == 2 and s.calls == 3  # one extraction, the merge tried twice
    once = graduate(
        store(tmp_path / "2"), seg(("a", "b")), Script([LOCK], [down, {"items": []}]), topic_embed
    )
    assert once.calls == 3 and len(once.items) == 2  # the second merge call answered


def test_model_answers_about_entities_merge_them(tmp_path: Path) -> None:
    k = store(tmp_path)
    k.add_entity(Entity(name="rate limiter", kind=EntityKind.COMPONENT, summary="redis lock"))
    extract = {
        "entities": [{"name": "throttle", "kind": "component", "summary": "redis lock limiter"}],
        "items": [
            {
                "type": "fact",
                "statement": "the throttle component keeps its counters in Redis",
                "entities": ["throttle"],
            }
        ],
    }
    ask = Script([extract], [{"items": [], "entities": [{"q": "Q1", "same_as": "C1"}]}])
    graduate(k, seg(("x", "y")), ask, topic_embed)
    assert "Q1: throttle" in ask.prompts[1]
    (ent,) = k.entities.values()
    assert ent.aliases == ["throttle"] and ent.mentions == 2
    (item,) = k.items.values()
    assert item.entities == [ent.entity_id]


def test_fuzzy_names() -> None:
    assert not fuzzy_safe("ui") and not fuzzy_safe("aaaaaaaa")
    assert fuzzy_safe("graph memory daemon")
    k = KnowledgeStore(Path("/nonexistent-gm-test"), None)
    e = k.add_entity(Entity(name="graph-memory daemon", kind=EntityKind.COMPONENT))
    assert k.fuzzy_entity("graph memory daemon") is e
    assert k.fuzzy_entity("graph memory server") is None


def test_prompt_and_chunks() -> None:
    big = "w " * 15000
    s = seg(("q1", big), ("q2", big), ("q3", "short"))
    parts = chunks(s, budget=40000)
    assert [len(c) for c in parts] == [1, 2]
    text = build_extract_prompt(s, parts[1], ["Redis"])
    assert "X2 (2026-11-01)" in text and "X3" in text and "X1 " not in text
    assert "Known entities: Redis" in text and "in the project limiter" in text


def test_segment_input_reads_buffered_exchanges() -> None:
    xs = [
        buffer.Exchange(
            prompt="fix the redis race",
            reply="lua script " * 400,
            files=[CodeAnchor(path="svc/limiter.py")],
            project="repo:/w/limiter",
        ),
        buffer.Exchange(prompt="and retries?", reply="backoff added", project="repo:/w/limiter"),
        buffer.Exchange(prompt="unrelated", reply="ok", project="repo:/w/other"),
    ]
    si = segment_input("s", xs, "idle", "earlier summary")
    assert [x.prompt for x in si.exchanges] == ["fix the redis race", "and retries?", "unrelated"]
    assert [x.id for x in si.exchanges] == [x.id for x in xs]
    assert "[…]" in si.exchanges[0].reply  # long replies keep their head and tail
    assert si.exchanges[0].files == ("svc/limiter.py",)
    assert (si.project, si.project_name) == ("repo:/w/limiter", "limiter")
    assert si.compact_summary == "earlier summary"
    assert segment_input("s", [xs[1].model_copy(update={"project": None})], "idle").project is None


def test_daemon_graduates_on_request(short_dir: Path) -> None:
    cfg = replace(load_config(short_dir), save_interval=0.05)
    daemon, t = _start(cfg, Script([LOCK]))
    try:
        call(cfg, "remember", {"session": "s", "turn": turn("add a lock", "done")}, autostart=False)
        assert daemon.buffer.sessions() == ["s"]
        res = call(cfg, "graduate", {"session": "s", "trigger": "session_end"}, autostart=False)
        assert res == {"queued": True}
        _wait(lambda: call(cfg, "status", autostart=False)["knowledge"]["segments"] == 1)
        st = call(cfg, "status", autostart=False)
        assert st["knowledge"]["items"] == 2 and st["knowledge"]["graduation"] is True
        _wait(lambda: daemon.buffer.sessions() == [])  # the ended session is gone
        view = call(cfg, "knowledge", autostart=False)
        some = view["items"][0]["id"]
        assert view["items"][0]["sources"][0]["session"] == "s"
        detail = call(cfg, "knowledge_item", {"id": some[:6]}, autostart=False)
        assert detail["item"]["item_id"] == some
        (learned,) = detail["learned"]
        assert (learned["session"], learned["trigger"], learned["exchanges"]) == (
            "s",
            "session_end",
            1,
        )
        assert (cfg.home / "memory.db").exists()  # saved before the buffer went
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_daemon_graduates_quiet_sessions(short_dir: Path) -> None:
    cfg = replace(load_config(short_dir), save_interval=0.05, graduation_idle=0.3)
    daemon, t = _start(cfg, Script([LOCK]))
    try:
        call(cfg, "remember", {"session": "s", "turn": turn("add a lock", "done")}, autostart=False)
        _wait(lambda: daemon.knowledge.stats()["segments"] == 1, timeout=8)
        (segment,) = daemon.knowledge.segments.values()
        assert segment.trigger == "idle"
        # the session may go on: its exchange became a summary of what was learned
        _wait(lambda: not daemon.buffer.exchanges("s"))
        (summary,) = daemon.buffer.summaries("s")
        assert summary.trigger == "idle" and summary.segment == segment.segment_id
        assert "SETNX" in summary.text and summary.items == segment.items
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_same_names_and_hubs(tmp_path: Path) -> None:
    k = store(tmp_path)
    extract = {
        "entities": [
            {"name": "graph_memory daemon", "kind": "component", "summary": ""},
            {"name": "graph-memory daemon", "kind": "component", "summary": ""},
            {"name": "the user", "kind": "person", "summary": ""},
            {"name": "user", "kind": "person", "summary": ""},
        ],
        "items": [
            {
                "type": "preference",
                "statement": "the user wants simple designs first",
                "entities": ["user"],
            },
            {
                "type": "fact",
                "statement": "the daemon is the only writer of the graph",
                "entities": ["graph-memory daemon", "the user"],
            },
        ],
    }
    graduate(k, seg(("a", "b")), Script([extract]), topic_embed)
    assert sorted(e.name for e in k.entities.values()) == ["graph_memory daemon", "the user"]
    assert k.entity_by_name("graph memory daemon") is not None
    # sharing only a person entity is no reason to link two items
    assert not k.edges


def test_short_file_names_join_their_paths(tmp_path: Path) -> None:
    k = store(tmp_path)
    extract = {
        "entities": [
            {"name": "src/pkg/engine.py", "kind": "file", "summary": ""},
            {"name": "engine.py", "kind": "file", "summary": ""},
        ],
        "items": [
            {
                "type": "fact",
                "statement": "engine.py is not thread safe at all",
                "entities": ["engine.py"],
            }
        ],
    }
    graduate(k, seg(("a", "b")), Script([extract]), topic_embed)
    (e,) = k.entities.values()
    assert e.name == "src/pkg/engine.py"
    assert next(iter(k.items.values())).entities == [e.entity_id]
    assert k.entity_by_name("engine.py") is e


# ---------------------------------------------------------------- Q&A, origins, summaries

SQLITE = {
    "entities": [{"name": "SQLite", "kind": "tool", "summary": "embedded SQL database"}],
    "items": [
        {
            "type": "learned",
            "statement": "SQLite has no array column type, so a recipe's tags go in a recipe_tag "
            "join table",
            "entities": ["SQLite"],
            "scope": "project",
            "evidence": ["X1"],
            "origin": "assistant",
            "applies_to": "SQLite 3",
            "volatility": "versioned",
        },
        {
            "type": "open",
            "statement": "the user is weighing option A over option B for recipe tags in SQLite, "
            "leaning A",
            "entities": ["SQLite"],
            "scope": "project",
            "origin": "nonsense",  # unknown values fall back
            "volatility": "sometimes",
        },
    ],
    "summary": "Designing a SQLite schema for recipes and tags; leaning to option A.",
}


def test_learned_items_keep_where_they_came_from(tmp_path: Path) -> None:
    from graph_memory.knowledge import Origin, Volatility

    k = KnowledgeStore(tmp_path, topic_embed)
    s = seg(("how do I model many-to-many in sqlite?", "Use a join table ..."))
    graduate(k, s, Script([SQLITE]), topic_embed)
    learned = next(i for i in k.items.values() if i.type is ItemType.LEARNED)
    assert (learned.origin, learned.applies_to, learned.volatility) == (
        Origin.ASSISTANT,
        "SQLite 3",
        Volatility.VERSIONED,
    )
    weighing = next(i for i in k.items.values() if i.type is ItemType.OPEN)
    assert (weighing.origin, weighing.volatility) == (Origin.ASSISTANT, Volatility.STABLE)

    # the same knowledge read in the docs later: reinforced, and now from docs
    docs = json.loads(json.dumps(SQLITE))
    docs["items"] = [
        {
            **SQLITE["items"][0],
            "origin": "docs",
            "ref": "https://www.sqlite.org/lang_createtable.html",
        }
    ]
    graduate(
        k,
        seg(("check the sqlite docs on tables", "The docs say ..."), start=T0 + timedelta(days=1)),
        Script([docs]),
        topic_embed,
    )
    assert learned.origin is Origin.DOCS and learned.ref.endswith("lang_createtable.html")
    assert learned.strength > 1.0 and learned.applies_to == "SQLite 3"
    # an assistant's restatement never downgrades it
    graduate(
        k,
        seg(("remind me about sqlite join tables", "..."), start=T0 + timedelta(days=2)),
        Script([SQLITE]),
        topic_embed,
    )
    assert learned.origin is Origin.DOCS


def test_the_extraction_prompt_carries_the_earlier_summary_and_asks_for_a_new_one() -> None:
    s = replace(
        seg(("and option A needs a join?", "Yes, through recipe_tag.")),
        earlier="Designing a SQLite schema for recipes and tags.",
    )
    p = build_extract_prompt(s, chunks(s)[0], [])
    assert "<earlier>\nDesigning a SQLite schema for recipes and tags.\n</earlier>" in p
    assert (
        '"summary": "<2-4 sentences>"' in p
        and '"origin": "<user|docs|code|observed|assistant>"' in p
    )
    assert "explanations of general knowledge" not in p  # Q&A is kept now
    read = buffer.Exchange(
        prompt="why are foreign keys off?",
        reply="The docs say so.",
        web=["https://www.sqlite.org/foreignkeys.html", "search: sqlite foreign keys"],
    )
    s2 = segment_input("s", [read], "idle")
    p2 = build_extract_prompt(s2, chunks(s2)[0], [])
    assert (
        "[read on the web: https://www.sqlite.org/foreignkeys.html, search: sqlite foreign keys]"
        in p2
    )
    assert "<earlier>" not in build_extract_prompt(seg(("a", "b")), chunks(seg(("a", "b")))[0], [])


def test_each_chunk_reads_the_summary_the_previous_chunk_wrote() -> None:
    from graph_memory.graduation import extract_step

    long = "x" * 30000
    s = replace(
        seg(("first question " + long, "a"), ("second question " + long, "b")), earlier="before"
    )
    out = [
        {"entities": [], "items": [], "summary": "after the first chunk"},
        {"entities": [], "items": [], "summary": "after both chunks"},
    ]
    script = Script(out)
    ex = extract_step(s, script, None, [])
    assert len(script.prompts) == 2
    assert "<earlier>\nbefore\n</earlier>" in script.prompts[0]
    assert "<earlier>\nafter the first chunk\n</earlier>" in script.prompts[1]
    assert ex.summary == "after both chunks"


def test_the_merge_prompt_marks_unchecked_and_disputed_items(tmp_path: Path) -> None:
    from graph_memory.graduation import extract_step, plan_merge
    from graph_memory.knowledge import Item, Origin

    k = KnowledgeStore(tmp_path, topic_embed)
    sqlite = k.add_entity(Entity(name="SQLite", kind=EntityKind.TOOL))
    old = k.add_item(
        Item(
            type=ItemType.LEARNED,
            statement="SQLite enforces foreign keys by default",
            entities=[sqlite.entity_id],
            origin=Origin.ASSISTANT,
        )
    )
    k.dispute(old.item_id, "the docs say they are off by default")
    s = seg(("are foreign keys on?", "The docs say they are off until a PRAGMA."))
    new = {
        "entities": [{"name": "SQLite", "kind": "tool"}],
        "items": [
            {
                "type": "learned",
                "statement": "SQLite foreign keys are off until PRAGMA foreign_keys = ON",
                "entities": ["SQLite"],
                "origin": "docs",
            }
        ],
    }
    plan = plan_merge(k, s, extract_step(s, Script([new]), topic_embed, []))
    assert plan.prompt is not None
    assert "N1 [learned, from docs] SQLite foreign keys are off until" in plan.prompt
    assert "[learned, 20" in plan.prompt
    assert (
        ", from the assistant] SQLite enforces foreign keys by default "
        "(DISPUTED: the docs say they are off by default)" in plan.prompt
    )


def test_an_answer_that_does_not_parse_is_asked_again(tmp_path: Path) -> None:
    import pytest

    from graph_memory.graduation import GraduationError, extract_step

    class Flaky(Script):
        def __init__(self, bad: int) -> None:
            super().__init__([{"entities": [], "items": [], "summary": "ok"}])
            self.bad = bad

        def __call__(self, prompt: str) -> str:
            if "distil" in prompt and self.bad:
                self.bad -= 1
                self.prompts.append(prompt)
                return '{"items": [{"statement": "broken "quote"}]}'
            return super().__call__(prompt)

    s = seg(("q", "a"))
    once = Flaky(1)
    ex = extract_step(s, once, None, [])
    assert ex.summary == "ok" and ex.calls == 2 and len(once.prompts) == 2
    with pytest.raises(GraduationError, match="bad JSON"):
        extract_step(s, Flaky(2), None, [])
