"""Where knowledge is kept: memory.db (SQLite, vectors searched by sqlite-vec), or the JSON
files from before it. A save writes only what changed, with its vectors, in one go."""

from __future__ import annotations

import random
from pathlib import Path

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
from graph_memory.storage import JsonBackend, SqliteBackend, differences
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
    assert again.items[a.item_id].statement == a.statement and b.item_id not in again.items
    assert [d["reason"] for d in again.deleted()] == ["noise"]
    assert again.summaries() == {"repo:/w/x": {"overview": "a limiter"}}
    assert again.vectors.get(a.item_id) is not None
    again.undelete(again.deleted()[0]["batch"])
    again.save()
    assert b.item_id in KnowledgeStore(tmp_path, topic_embed).items


def test_a_save_writes_only_what_changed(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    for n in range(30):
        k.add_item(Item(type=ItemType.FACT, statement=f"fact number {n}"))
    k.save()
    assert isinstance(k.backend, SqliteBackend)
    conn = k.backend.conn
    one = next(iter(k.items))
    before = conn.total_changes
    k.mark_used([one])  # one item: stronger, used once more
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
    # another reader (a crash would leave the same) sees the saved state only: the new
    # item's vector is not there without its row, or the other way round
    other = KnowledgeStore(tmp_path, topic_embed)
    assert kept.item_id in other.items and lost.item_id not in other.items
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
    what it finds stays what the in-memory index finds among the visible notes, as notes
    change scope, become invalid, are deleted and come back."""
    from graph_memory.knowledge_recall import KnowledgeIndex

    rnd = random.Random(5)
    scopes = ["global", "repo:/a", "repo:/b", "topic:t"]
    db = KnowledgeStore(tmp_path / "db")
    mem = KnowledgeStore(tmp_path / "mem", backend=JsonBackend(tmp_path / "mem"))
    for n in range(300):
        v = [rnd.uniform(-1, 1) for _ in range(16)]
        it = Item(type=ItemType.FACT, statement=f"note {n}", scope=rnd.choice(scopes))
        db.add_item(it.model_copy(), vector=v)
        mem.add_item(it.model_copy(), vector=v)
    ids = list(db.items)

    def same(here: list[str]) -> None:
        index = {}
        for k in (db, mem):
            index[k] = KnowledgeIndex(k)
            index[k].fresh()
        visible = index[mem].visible(here)
        assert index[db].visible(here) == visible and len(visible) > 50
        for _ in range(5):
            q = [rnd.uniform(-1, 1) for _ in range(16)]
            got = db.similar(q, "item", 12, -1.0, among=visible, scopes=here)
            want = mem.similar(q, "item", 12, -1.0, among=visible)
            assert len(want) == 12 and [i for i, _ in got] == [i for i, _ in want]

    same(["global", "repo:/a"])
    for k in (db, mem):  # many at once: filed from one scan
        for i in ids[:100]:
            k.touch(i)
            k.items[i].scope = "repo:/a"  # moved after being marked: filed at the search
        for i in ids[100:120]:
            k.touch(i)
            k.items[i].invalid_at = T0
    same(["global", "repo:/a"])
    db.save()
    db.close()
    db = KnowledgeStore(tmp_path / "db")  # what was filed is on disk
    same(["global", "repo:/a"])
    same(["repo:/b", "topic:t"])
    for k in (db, mem):  # a few: looked up one by one
        for i in ids[:10]:
            k.touch(i)
            k.items[i].scope = "topic:t"
    same(["repo:/b", "topic:t"])
    for k in (db, mem):
        k.delete(ids[60:90], "noise")
    same(["global", "repo:/a", "repo:/b"])
    for k in (db, mem):
        k.undelete(k.deleted()[0]["batch"])
    same(["global", "repo:/a", "repo:/b"])


def test_a_memory_from_before_sqlite_still_works_from_its_json_files(tmp_path: Path) -> None:
    old = KnowledgeStore(tmp_path, topic_embed, backend=JsonBackend(tmp_path))
    _memory(old)
    old.save()
    again = KnowledgeStore(tmp_path, topic_embed)  # the files are there: they are read
    assert again.backend.name == "json" and len(again.items) == 2
    assert not (tmp_path / "memory.db").exists()
    assert differences(old, again) == []


def test_a_python_that_cannot_load_sqlite_extensions_keeps_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "sqlite_problem", lambda: "no extensions here")
    k = KnowledgeStore(tmp_path / "new", topic_embed)
    assert k.backend.name == "json"
    with pytest.raises(storage.StorageError, match="no extensions here"):
        storage.connect(tmp_path / "x.db")
    assert "no extensions here" in (storage.json_note("json") or "")  # gm status says why
    monkeypatch.setattr(storage, "sqlite_problem", lambda: None)
    assert "gm migrate" in (storage.json_note("json") or "")
    assert storage.json_note("sqlite") is None


def test_differences_finds_what_differs(tmp_path: Path) -> None:
    a = KnowledgeStore(tmp_path / "a", topic_embed)
    b = KnowledgeStore(tmp_path / "b", topic_embed)
    x, _ = _memory(a)
    assert differences(a, b)  # b is empty
    b2 = KnowledgeStore(tmp_path / "c", topic_embed, backend=JsonBackend(tmp_path / "c"))
    a.copy_to(tmp_path / "copy")
    same = KnowledgeStore(tmp_path / "copy", topic_embed)
    assert differences(a, same) == []
    same.items[x.item_id].statement = "changed"
    assert differences(a, same) == [f"item: 1 differ, e.g. {x.item_id}"]
    assert b2.backend.name == "json"


# --------------------------------------------------------------------------- moving


def _json_memory(home: Path) -> KnowledgeStore:
    k = KnowledgeStore(home, topic_embed, backend=JsonBackend(home))
    a, b = _memory(k)
    k.add_item(Item(type=ItemType.FACT, statement="noise worth deleting"))
    k.delete([next(i for i in k.items if k.items[i].statement.startswith("noise"))], "noise")
    k.set_summaries({"repo:/w/x": {"overview": "a limiter"}})
    k.save()
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
    _json_memory(home).copy_to(tmp_path / "as-it-was")  # the move removes home's files
    before = KnowledgeStore(tmp_path / "as-it-was", topic_embed)
    monkeypatch.setenv("GM_HOME", str(home))
    assert cli.main(["migrate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 items" not in out and "2 items, 1 entities, 1 edges, 1 graduations" in out
    assert "1 deletions, 1 summaries" in out and not (home / "memory.db").exists()

    assert cli.main(["migrate"]) == 0
    assert "now in" in capsys.readouterr().out
    assert sorted(p.name for p in home.iterdir() if not p.name.startswith("memory.db-")) == [
        f"archive-{migrate.clock.now().astimezone().date().isoformat()}-json.tar.gz",
        "memory.db",
    ]
    after = KnowledgeStore(home, topic_embed)
    assert after.backend.name == "sqlite" and differences(before, after) == []
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
    exported = KnowledgeStore(tmp_path / "out", topic_embed)
    assert exported.backend.name == "json" and differences(before, exported) == []


def test_a_difference_stops_the_move_and_leaves_the_json_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from graph_memory import migrate

    _json_memory(tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(migrate, "differences", lambda a, b: ["item: 1 differ"])
    with pytest.raises(RuntimeError, match="differs from the JSON files"):
        migrate.sqlite_apply(migrate.sqlite_plan(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == files  # nothing new, nothing gone
    assert KnowledgeStore(tmp_path).backend.name == "json"
