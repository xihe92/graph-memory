"""Where an item was learned: (session, newest exchange that taught it). Recall uses it to
skip items the session can still see."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from graph_memory.graduation import Exchange, SegmentInput, graduate
from graph_memory.knowledge import Item, ItemType, KnowledgeStore, Source, add_source, in_view

from .conftest import topic_embed
from .test_graduation import LOCK, Script

T0 = datetime(2026, 11, 1, 10, tzinfo=UTC)


def _item(*sources: tuple[str, datetime]) -> Item:
    return Item(
        type=ItemType.FACT,
        statement="the limiter uses a redis lua script",
        sources=[Source(session=s, until=u) for s, u in sources],
    )


def test_in_view_until_the_session_compacts_past_it() -> None:
    it = _item(("a", T0))
    assert in_view(it, "a", None)  # no compaction yet: the session still sees it
    assert in_view(it, "a", T0 - timedelta(minutes=5))  # compacted before it was taught
    assert not in_view(it, "a", T0 + timedelta(minutes=5))  # compacted away
    assert not in_view(it, "b", None)  # another session never saw it


def test_a_compaction_graduation_yields_items_to_show_again() -> None:
    """The compaction at 11:00 triggers graduation, so the item is learned at 11:01, after
    the compaction. It must still count as out of view: it was taught at 10:40, before."""
    compacted = T0 + timedelta(hours=1)
    it = _item(("a", T0 + timedelta(minutes=40)))
    it.created_at = compacted + timedelta(minutes=1)
    assert not in_view(it, "a", compacted)


def test_an_item_taught_by_two_sessions_is_never_in_view() -> None:
    it = _item(("a", T0), ("b", T0))
    assert not in_view(it, "a", None) and not in_view(it, "b", None)
    assert not in_view(Item(type=ItemType.FACT, statement="no sources at all"), "a", None)


def test_add_source_keeps_one_entry_per_session_with_the_newest_time() -> None:
    it = _item(("a", T0))
    add_source(it, "a", T0 - timedelta(hours=1))
    add_source(it, "a", T0 + timedelta(hours=1))
    add_source(it, "b", T0)
    assert [(s.session, s.until) for s in it.sources] == [
        ("a", T0 + timedelta(hours=1)),
        ("b", T0),
    ]


def test_graduation_records_sources_and_reinforcing_adds_one(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)

    def seg(session: str, start: datetime) -> SegmentInput:
        xs = (
            Exchange("x1", start, "add a lock", "done"),
            Exchange("x2", start + timedelta(minutes=1), "add retries", "done"),
        )
        return SegmentInput(session, xs, "session_end", "repo:/w/limiter", "limiter")

    graduate(k, seg("s1", T0), Script([LOCK]), topic_embed)
    lock = next(i for i in k.scan() if "SETNX" in i.statement)
    retries = next(i for i in k.scan() if "retries" in i.statement)
    # each item's time is the newest exchange that taught it (its evidence), not the end
    assert [(s.session, s.until) for s in lock.sources] == [("s1", T0)]
    assert retries.sources[0].until == T0 + timedelta(minutes=1)

    later = T0 + timedelta(days=1)
    graduate(k, seg("s2", later), Script([LOCK]), topic_embed)  # states both again
    assert [s.session for s in lock.sources] == ["s1", "s2"]


def test_items_from_before_sources_get_them_from_their_segments(tmp_path: Path) -> None:
    import sqlite3

    from graph_memory import migrate
    from graph_memory.migrate import export_json

    k = KnowledgeStore(tmp_path / "work", topic_embed)
    graduate(
        k,
        SegmentInput("s1", (Exchange("x1", T0, "add a lock", "done"),), "session_end"),
        Script([LOCK]),
        topic_embed,
    )
    k.save()
    # such items exist in the JSON files written before sources: `gm migrate` gives them
    # theirs on the way into memory.db
    old = tmp_path / "old"
    export_json(k, old)
    data = json.loads((old / "knowledge.json").read_text())
    for i in data["items"]:
        i.pop("sources")  # as written before this change
    (old / "knowledge.json").write_text(json.dumps(data))
    migrate.sqlite_apply(migrate.sqlite_plan(old))
    # and a memory.db holding one without them gets them when it is opened
    k.close()
    conn = sqlite3.connect(tmp_path / "work" / "memory.db")
    conn.execute("UPDATE item SET body = json_set(body, '$.sources', json('[]'))")
    conn.commit()
    conn.close()
    for home in (old, tmp_path / "work"):
        again = KnowledgeStore(home, topic_embed)
        (seg,) = again.segments()
        for it in again.scan():
            assert [(s.session, s.until) for s in it.sources] == [("s1", seg.end)]
            assert again.heads[it.item_id].sessions == ("s1",)
        assert seg.end == T0
