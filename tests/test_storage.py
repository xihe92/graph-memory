"""Where knowledge is kept: memory.db (SQLite, vectors searched by sqlite-vec). The store
keeps only small fields in memory and reads the rest by id; a save writes only what
changed, with its vectors and words, in one go. The JSON files from before SQLite are
read by `gm migrate` and written by `gm export`."""

from __future__ import annotations

import random
import sqlite3
import tracemalloc
from pathlib import Path

import pydantic
import pytest

from graph_memory import storage
from graph_memory.knowledge import (
    Entity,
    EntityKind,
    Item,
    ItemType,
    KnowledgeStore,
    Relation,
    Segment,
)
from graph_memory.migrate import export_json
from graph_memory.storage import SqliteBackend, differences, read_db, read_json
from graph_memory.vectors import JsonVectorIndex

from .conftest import topic_embed
from .test_graduation import T0


def _memory(k: KnowledgeStore) -> tuple[Item, Item]:
    redis = k.add_entity(Entity(name="Redis", kind=EntityKind.TOOL))
    a = k.add_item(
        Item(
            type=ItemType.DECISION,
            statement="the limiter takes a redis lock",
            entities=[redis.entity_id],
        )
    )
    b = k.add_item(Item(type=ItemType.PITFALL, statement="the lock must expire", scope="repo:/w/x"))
    k.add_edge(a.item_id, b.item_id, Relation.DEPENDS_ON, 0.9, reason="needs it")
    k.add_segment(Segment(session="s1", start=T0, end=T0, trigger="session_end", items=[a.item_id]))
    return a, b


def test_a_new_memory_is_one_sqlite_file_that_reads_back_the_same(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    assert k.backend.name == "sqlite"
    a, b = _memory(k)
    k.set_summaries({"repo:/w/x": {"overview": "a limiter"}})
    k.delete([b.item_id], "noise")
    k.save()
    k.close()
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith("memory.db-")) == [
        "memory.db"
    ]
    again = KnowledgeStore(tmp_path, topic_embed)
    got = again.item(a.item_id)
    assert got is not None and got.statement == a.statement and not again.has(b.item_id)
    assert [d["reason"] for d in again.deleted()] == ["noise"]
    assert again.summaries() == {"repo:/w/x": {"overview": "a limiter"}}
    assert again.vectors.get(a.item_id) is not None
    again.undelete(again.deleted()[0]["batch"])
    again.save()
    assert KnowledgeStore(tmp_path, topic_embed).has(b.item_id)


def test_only_small_fields_stay_in_memory_and_a_read_copy_cannot_be_changed(
    tmp_path: Path,
) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a, _ = _memory(k)
    k.save()
    assert not k._items and not k._entities and not k._edges and not k._segments
    with pytest.raises(pydantic.ValidationError):
        a.strength = 5.0  # saved: changing the old object would be lost, so it fails
    read = k.item(a.item_id)
    assert read is not None
    with pytest.raises(pydantic.ValidationError):
        read.scope = "global"  # a read copy: edit() gives one to change
    k.edit(a.item_id).scope = "repo:/w/y"
    k.touch(a.item_id)
    assert k.heads[a.item_id].scope == "repo:/w/y"
    k.save()
    again = KnowledgeStore(tmp_path, topic_embed)
    assert again.heads == k.heads and again.names == k.names and again.links == k.links
    assert again.differences(k) == []


def test_a_save_writes_only_what_changed(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    for n in range(30):
        k.add_item(Item(type=ItemType.FACT, statement=f"fact number {n}"))
    k.save()
    conn = k.backend.conn
    one = next(iter(k.heads))
    before = conn.total_changes
    k.mark_used([one])  # one item: stronger, used once more; its words are the same
    k.save()
    assert conn.total_changes - before == 1  # its row, and nothing else
    before = conn.total_changes
    k.save()
    assert conn.total_changes == before  # nothing changed, nothing written


def test_what_was_not_saved_is_not_on_disk_vectors_included(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    kept = k.add_item(Item(type=ItemType.FACT, statement="saved redis note"))
    k.save()
    lost = k.add_item(Item(type=ItemType.FACT, statement="never saved redis note"))
    k.flush()  # written into the open transaction, not committed
    # another reader (a crash would leave the same) sees the saved state only: the new
    # item's vector is not there without its row, or the other way round
    other = KnowledgeStore(tmp_path, topic_embed)
    assert other.has(kept.item_id) and not other.has(lost.item_id)
    assert other.vectors.get(kept.item_id) is not None and other.vectors.get(lost.item_id) is None
    other.close()


def test_sqlite_search_gives_what_the_in_memory_index_gives(tmp_path: Path) -> None:
    rnd = random.Random(3)
    mem = JsonVectorIndex(tmp_path / "v.json")
    k = KnowledgeStore(tmp_path / "db")
    db = k.vectors
    for n in range(200):
        v = [rnd.uniform(-1, 1) for _ in range(16)]
        kind = "item" if n % 3 else "entity"
        mem.upsert(f"n{n}", v, {"kind": kind})
        db.upsert(f"n{n}", v, {"kind": kind})
    db.remove("n7")
    mem.remove("n7")
    some = {f"n{n}" for n in range(0, 200, 5)}
    for _ in range(20):
        q = [rnd.uniform(-1, 1) for _ in range(16)]
        for kw in (
            {},
            {"where": {"kind": "item"}},
            {"among": some},
            {"exclude": frozenset({"n1", "n2"})},
        ):
            want = mem.query(q, 12, **kw)  # type: ignore[arg-type]
            got = db.query(q, 12, **kw)  # type: ignore[arg-type]
            assert [i for i, _ in got] == [i for i, _ in want], kw
            assert all(abs(s - w) < 1e-5 for (_, s), (_, w) in zip(got, want, strict=True))
    # a note in the set with no vector does not shorten the answer
    assert len(db.query(q, 12, among=some | {"no-vector"})) == len(mem.query(q, 12, among=some))
    assert db.get("n7") is None and db.get("n8") == pytest.approx(mem.get("n8"), abs=1e-6)


def test_a_scoped_search_follows_notes_that_move_or_fade(tmp_path: Path) -> None:
    """The database filters a recall's search by scope and recallability inside the scan;
    what it finds stays what an exact search over the visible notes finds, as notes
    change scope, become invalid, are deleted and come back."""
    from graph_memory.knowledge_recall import KnowledgeIndex

    rnd = random.Random(5)
    scopes = ["global", "repo:/a", "repo:/b", "topic:t"]
    db = KnowledgeStore(tmp_path / "db")
    exact = JsonVectorIndex(tmp_path / "v.json")
    for n in range(300):
        v = [rnd.uniform(-1, 1) for _ in range(16)]
        it = db.add_item(
            Item(type=ItemType.FACT, statement=f"note {n}", scope=rnd.choice(scopes)), vector=v
        )
        exact.upsert(it.item_id, v, {"kind": "item"})
    ids = list(db.heads)

    def same(here: list[str]) -> None:
        index = KnowledgeIndex(db)
        index.fresh()
        visible = index.visible(here)
        want_visible = {i for i, h in db.heads.items() if h.recallable and h.scope in here}
        assert visible == want_visible and len(visible) > 50
        for _ in range(5):
            q = [rnd.uniform(-1, 1) for _ in range(16)]
            got = db.similar(q, "item", 12, -1.0, among=visible, scopes=here)
            want = exact.query(q, 12, among=visible)
            assert len(want) == 12 and [i for i, _ in got] == [i for i, _ in want]

    same(["global", "repo:/a"])
    for i in ids[:100]:  # many at once: filed from one scan
        db.edit(i).scope = "repo:/a"
        db.touch(i)
    for i in ids[100:120]:
        db.edit(i).invalid_at = T0
        db.touch(i)
    same(["global", "repo:/a"])
    db.save()
    db.close()
    db = KnowledgeStore(tmp_path / "db")  # what was filed is on disk
    same(["global", "repo:/a"])
    same(["repo:/b", "topic:t"])
    for i in ids[:10]:  # a few: looked up one by one
        db.edit(i).scope = "topic:t"
        db.touch(i)
    same(["repo:/b", "topic:t"])
    db.delete(ids[60:90], "noise")
    for i in ids[60:90]:
        exact.remove(i)
    same(["global", "repo:/a", "repo:/b"])
    batch = db.undelete(db.deleted()[0]["batch"])
    for rec in batch["items"]:
        exact.upsert(rec["item"]["item_id"], rec["vector"], {"kind": "item"}, unit=True)
    same(["global", "repo:/a", "repo:/b"])


def test_a_memory_still_in_json_files_asks_for_gm_migrate(tmp_path: Path) -> None:
    src = KnowledgeStore(tmp_path / "src", topic_embed)
    _memory(src)
    src.save()
    export_json(src, tmp_path / "old")
    with pytest.raises(storage.StorageError, match="gm migrate"):
        KnowledgeStore(tmp_path / "old", topic_embed)
    assert not (tmp_path / "old" / "memory.db").exists()


def test_a_python_that_cannot_load_sqlite_extensions_is_told_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "sqlite_problem", lambda: "no extensions here")
    with pytest.raises(storage.StorageError, match="no extensions here"):
        KnowledgeStore(tmp_path / "new", topic_embed)


def test_an_older_memory_db_gets_its_word_index_once(tmp_path: Path) -> None:
    """A memory.db from before the word index (schema 1) is brought up to date when it is
    first opened; recall then finds its notes by their words."""
    from graph_memory.knowledge_recall import KnowledgeIndex

    k = KnowledgeStore(tmp_path, topic_embed)
    a, _ = _memory(k)
    k.save()
    k.close()
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.executescript(
        "DROP TABLE word; DROP TABLE doc; DELETE FROM meta WHERE key = 'words';"
        " UPDATE meta SET value = '1' WHERE key = 'schema';"
    )
    conn.close()
    again = KnowledgeStore(tmp_path, topic_embed)
    ix = KnowledgeIndex(again)
    assert [i for i, _, _ in ix.bm25(["limiter", "redis"])] == [a.item_id]
    again.save()
    again.close()
    conn = sqlite3.connect(tmp_path / "memory.db")
    assert conn.execute("SELECT value FROM meta WHERE key = 'words'").fetchone() == ("1",)
    assert conn.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("3",)
    conn.close()
    assert KnowledgeStore(tmp_path, topic_embed).word_differences() == []


def test_differences_finds_what_differs(tmp_path: Path) -> None:
    a = KnowledgeStore(tmp_path / "a", topic_embed)
    b = KnowledgeStore(tmp_path / "b", topic_embed)
    x, _ = _memory(a)
    a.save()
    assert differences(read_db(a.backend), read_db(b.backend))  # b is empty
    a.copy_to(tmp_path / "copy")
    same = KnowledgeStore(tmp_path / "copy", topic_embed)
    assert differences(read_db(a.backend), read_db(same.backend)) == []
    assert a.differences(same) == []
    same.edit(x.item_id).statement = "changed"
    same.touch(x.item_id)
    same.save()
    assert differences(read_db(a.backend), read_db(same.backend)) == [
        f"item: 1 differ, e.g. {x.item_id}"
    ]
    assert a.differences(same)  # its head: the statement hash and length


# --------------------------------------------------------------------------- moving


def test_an_older_memory_db_gets_its_active_days_once(tmp_path: Path) -> None:
    """A memory.db from before active days were kept (schema 2) gets them, when first
    opened, from the dates it records work on: its graduations and its notes' times."""
    from graph_memory.active_days import local_date

    k = KnowledgeStore(tmp_path, topic_embed)
    assert k.days.rows() == []  # a new memory: nothing to fill in
    a, b = _memory(k)  # a segment at T0 that taught a; b, a project note, learned now
    k.save()
    k.close()
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.executescript(
        "DROP TABLE active_day; DELETE FROM meta WHERE key = 'days';"
        " UPDATE meta SET value = '2' WHERE key = 'schema';"
    )
    conn.close()
    again = KnowledgeStore(tmp_path, topic_embed)
    t0, today = local_date(T0).isoformat(), local_date(b.created_at).isoformat()
    want = {(t0, "global"), (today, "global"), (today, "repo:/w/x")}
    assert set(again.days.rows()) == want
    again.save()
    again.close()
    conn = sqlite3.connect(tmp_path / "memory.db")
    assert set(conn.execute("SELECT date, scope FROM active_day")) == want
    assert conn.execute("SELECT value FROM meta WHERE key = 'days'").fetchone() == ("1",)
    conn.execute("DELETE FROM active_day")
    conn.commit()
    conn.close()
    assert KnowledgeStore(tmp_path, topic_embed).days.rows() == []  # filled once only


def _json_memory(home: Path, work: Path) -> KnowledgeStore:
    """A memory in the JSON files from before SQLite, in `home` (written by `gm export`
    from one built in `work`)."""
    k = KnowledgeStore(work, topic_embed)
    _memory(k)
    noise = k.add_item(Item(type=ItemType.FACT, statement="noise worth deleting"))
    k.delete([noise.item_id], "noise")
    k.set_summaries({"repo:/w/x": {"overview": "a limiter"}})
    k.mark_active("repo:/w/x", T0)
    k.save()
    export_json(k, home)
    return k


def test_gm_migrate_moves_json_into_sqlite_and_keeps_the_files(
    tmp_path: Path,
    short_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tarfile

    from graph_memory import cli, migrate

    home = short_dir  # short: the socket path `gm` checks for a running daemon must fit
    _json_memory(home, tmp_path / "work")
    before = read_json(home)
    monkeypatch.setenv("GM_HOME", str(home))
    assert cli.main(["migrate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 items" not in out and "2 items, 1 entities, 1 edges, 1 graduations" in out
    assert "1 deletions, 1 summaries, 2 active days" in out
    assert not (home / "memory.db").exists()

    assert cli.main(["migrate"]) == 0
    assert "now in" in capsys.readouterr().out
    assert sorted(p.name for p in home.iterdir() if not p.name.startswith("memory.db-")) == [
        f"archive-{migrate.clock.now().astimezone().date().isoformat()}-json.tar.gz",
        "memory.db",
    ]
    after = KnowledgeStore(home, topic_embed)
    assert after.backend.name == "sqlite" and differences(before, read_db(after.backend)) == []
    assert after.word_differences() == []  # its word index was built by the move
    assert len(after.days.rows()) == 2  # its active days came along
    with tarfile.open(next(home.glob("archive-*-json.tar.gz"))) as tar:
        assert set(tar.getnames()) == {
            "knowledge.json",
            "knowledge_vectors.json",
            "knowledge_deleted.jsonl",
            "summaries.json",
        }
    after.close()
    assert cli.main(["migrate"]) == 0  # nothing left to do
    assert "already in" in capsys.readouterr().out

    # and back out: the export reads as the same memory
    assert cli.main(["export", str(tmp_path / "out")]) == 0
    assert differences(before, read_json(tmp_path / "out")) == []


def test_a_difference_stops_the_move_and_leaves_the_json_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from graph_memory import migrate

    home = tmp_path / "home"
    _json_memory(home, tmp_path / "work")
    files = sorted(p.name for p in home.iterdir())
    monkeypatch.setattr(migrate, "differences", lambda a, b: ["item: 1 differ"])
    with pytest.raises(RuntimeError, match="differs from the JSON files"):
        migrate.sqlite_apply(migrate.sqlite_plan(home))
    assert sorted(p.name for p in home.iterdir()) == files  # nothing new, nothing gone
    with pytest.raises(storage.StorageError, match="gm migrate"):
        KnowledgeStore(home)


@pytest.mark.slow
def test_memory_held_at_ten_thousand_notes_stays_small(tmp_path: Path) -> None:
    """What the store and the recall index hold in Python, at 10,000 notes with 20,000
    links: a few megabytes, not the notes themselves."""
    from graph_memory.knowledge_recall import KnowledgeIndex

    rnd = random.Random(11)
    words = [f"w{n}" for n in range(3000)]
    k = KnowledgeStore(tmp_path)
    ents = [
        k.add_entity(Entity(name=f"module{n}/file{n}.py", kind=EntityKind.FILE), vector=[1.0, 0.0])
        for n in range(2000)
    ]
    for _ in range(10_000):
        k.add_item(
            Item(
                type=ItemType.FACT,
                statement=" ".join(rnd.choices(words, k=25)),
                why=" ".join(rnd.choices(words, k=12)),
                scope=rnd.choice(["global", "repo:/a", "repo:/b", "topic:t"]),
                entities=[e.entity_id for e in rnd.sample(ents, 2)],
            ),
            vector=[rnd.uniform(-1, 1), rnd.uniform(-1, 1)],
        )
    ids = list(k.heads)
    for _ in range(20_000):
        a, b = rnd.sample(ids, 2)
        k.add_edge(a, b, Relation.SAME_AREA, 0.6)
    k.save()
    k.close()
    tracemalloc.start()
    store = KnowledgeStore(tmp_path)
    index = KnowledgeIndex(store)
    index.fresh()
    held = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert len(store.heads) == 10_000 and not store._items
    assert held < 30e6, f"{held / 1e6:.1f} MB"
    assert isinstance(store.backend, SqliteBackend)
