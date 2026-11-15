"""Where knowledge is kept on disk: one SQLite file, or (before `gm migrate`) JSON files.

    memory.db      items, entities, edges, graduations, deletions, summaries, and every
                   vector (searched in place by sqlite-vec), in one file. A save writes
                   only the rows that changed, with their vectors, in one transaction.
    knowledge.json + knowledge_vectors.json + knowledge_deleted.jsonl + summaries.json
                   the format before SQLite: still read, so a memory keeps working until
                   it is migrated, and written by `gm export`.

The store (knowledge.py) keeps everything it needs in memory and tells its backend what
changed; a backend only reads and writes.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .vectors import JsonVectorIndex, SqliteVectorIndex, VectorIndex, atomic_write

if TYPE_CHECKING:
    from .knowledge import KnowledgeStore

DB_FILE = "memory.db"
KNOWLEDGE_FILE = "knowledge.json"
VECTORS_FILE = "knowledge_vectors.json"
DELETED_FILE = "knowledge_deleted.jsonl"
SUMMARY_FILE = "summaries.json"
JSON_FILES = (KNOWLEDGE_FILE, VECTORS_FILE, DELETED_FILE, SUMMARY_FILE)
KNOWLEDGE_SCHEMA = 1
DB_SCHEMA = 1

Rows = dict[str, list[dict[str, Any]]]  # "entities", "items", "edges", "segments"


class StorageError(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    @property
    def vectors(self) -> VectorIndex: ...

    def load(self) -> Rows: ...
    def save(self, store: KnowledgeStore, full: bool) -> None: ...
    def log_deletion(self, batch: dict[str, Any]) -> None: ...
    def deletions(self) -> list[dict[str, Any]]: ...
    def drop_deletion(self, batch_id: str) -> None: ...
    def summaries(self) -> dict[str, Any]: ...
    def set_summaries(self, data: dict[str, Any]) -> None: ...
    def copy_to(self, dest: Path) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------- SQLite


def sqlite_problem() -> str | None:
    """Why this Python cannot use the SQLite storage, or None when it can."""
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return (
            "this Python's sqlite3 module cannot load SQLite extensions (pyenv's default "
            "build and the macOS system Python leave it out); install graph-memory on "
            "uv's own Python: uv tool install --managed-python --editable <repo> --force"
        )
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return (
            "the sqlite-vec package is not installed; reinstall graph-memory: "
            "uv tool install --managed-python --editable <repo> --force"
        )
    return None


def json_note(backend: str) -> str | None:
    """For a memory kept in JSON files: why, and what moves it to memory.db."""
    if backend != "json":
        return None
    problem = sqlite_problem()
    if problem:
        return f"memory is in JSON files because SQLite storage cannot be used: {problem}"
    return (
        "memory is in the JSON files from before SQLite; `gm stop`, then `gm migrate`, "
        "moves it to memory.db"
    )


def connect(path: Path) -> sqlite3.Connection:
    """A connection with sqlite-vec loaded, in autocommit mode: SqliteBackend opens and
    commits its transactions itself."""
    problem = sqlite_problem()
    if problem:
        raise StorageError(problem)
    import sqlite_vec

    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")  # another reader or writer: wait, don't fail
    return conn


_TABLES = {
    # kind: (table, key columns, extra columns kept beside the JSON body for SQL to filter)
    "item": ("item", ("id",), ("type", "scope", "valid")),
    "entity": ("entity", ("id",), ("name", "kind")),
    "edge": ("edge", ("source", "target", "relation"), ()),
    "segment": ("segment", ("id",), ("session",)),
}


class SqliteBackend:
    """memory.db. The store's changes are written in an open transaction (vectors as
    soon as they change, rows at save) and committed by save(), so notes and their
    vectors reach disk together or not at all."""

    name = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = connect(path)
        c = self.conn
        # set up a new file; an existing one is only read here, so opening it never waits
        # for (or blocks) the daemon's writes
        new = c.execute("SELECT 1 FROM sqlite_master WHERE name = 'item'").fetchone() is None
        if new:
            c.execute("BEGIN IMMEDIATE")
            c.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS item(id TEXT PRIMARY KEY, type TEXT, scope TEXT,"
                " valid INTEGER, body TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS entity(id TEXT PRIMARY KEY, name TEXT, kind TEXT,"
                " body TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS edge(source TEXT, target TEXT, relation TEXT,"
                " body TEXT NOT NULL, PRIMARY KEY (source, target, relation))"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS segment(id TEXT PRIMARY KEY, session TEXT,"
                " body TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS deleted(batch TEXT PRIMARY KEY, at TEXT, body TEXT)"
            )
            c.execute("CREATE TABLE IF NOT EXISTS summary(scope TEXT PRIMARY KEY, body TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS item_scope ON item(scope)")
            c.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(DB_SCHEMA),))
            c.execute("COMMIT")
        self.vectors = SqliteVectorIndex(c, self.begin)

    def begin(self) -> None:
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN")

    def commit(self) -> None:
        if self.conn.in_transaction:
            self.conn.execute("COMMIT")

    def load(self) -> Rows:
        out: Rows = {}
        for kind, (table, _, _) in _TABLES.items():
            rows = self.conn.execute(f"SELECT body FROM {table} ORDER BY rowid")  # noqa: S608
            out[kind + "s" if kind != "entity" else "entities"] = [json.loads(b) for (b,) in rows]
        return out

    def _write(self, kind: str, obj: Any) -> None:
        table, keys, extra = _TABLES[kind]
        body = obj.model_dump_json()
        vals: tuple[Any, ...]
        if kind == "item":
            vals = (obj.item_id, obj.type.value, obj.scope, int(obj.valid), body)
        elif kind == "entity":
            vals = (obj.entity_id, obj.name, obj.kind.value, body)
        elif kind == "edge":
            vals = (obj.source, obj.target, obj.relation.value, body)
        else:
            vals = (obj.segment_id, obj.session, body)
        marks = ", ".join("?" * len(vals))
        self.conn.execute(f"INSERT OR REPLACE INTO {table} VALUES ({marks})", vals)  # noqa: S608

    def _remove(self, kind: str, key: Any) -> None:
        table, keys, _ = _TABLES[kind]
        where = " AND ".join(f"{k} = ?" for k in keys)
        args = key if isinstance(key, tuple) else (key,)
        self.conn.execute(f"DELETE FROM {table} WHERE {where}", args)  # noqa: S608

    def save(self, store: KnowledgeStore, full: bool) -> None:
        self.begin()
        try:
            for kind, live in store.collections().items():
                if full:
                    table, keys, _ = _TABLES[kind]
                    cols = ", ".join(keys)
                    on_disk = {
                        tuple(r) if len(r) > 1 else r[0]
                        for r in self.conn.execute(f"SELECT {cols} FROM {table}")  # noqa: S608
                    }
                    for key in on_disk - set(live):
                        self._remove(kind, key)
                    for obj in live.values():
                        self._write(kind, obj)
                    continue
                for key in store.changed(kind):
                    if key in live:
                        self._write(kind, live[key])
                    else:
                        self._remove(kind, key)
            self.commit()
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def log_deletion(self, batch: dict[str, Any]) -> None:
        self.begin()  # committed with the save that follows the deletion
        self.conn.execute(
            "INSERT OR REPLACE INTO deleted VALUES (?, ?, ?)",
            (batch["batch"], batch["at"], json.dumps(batch)),
        )

    def deletions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT body FROM deleted ORDER BY at, rowid")
        return [json.loads(b) for (b,) in rows]

    def drop_deletion(self, batch_id: str) -> None:
        self.begin()
        self.conn.execute("DELETE FROM deleted WHERE batch = ?", (batch_id,))

    def summaries(self) -> dict[str, Any]:
        return {s: json.loads(b) for s, b in self.conn.execute("SELECT scope, body FROM summary")}

    def set_summaries(self, data: dict[str, Any]) -> None:
        self.begin()  # committed by the store's save, with everything else
        self.conn.execute("DELETE FROM summary")
        self.conn.executemany(
            "INSERT INTO summary VALUES (?, ?)", [(s, json.dumps(b)) for s, b in data.items()]
        )

    def copy_to(self, dest: Path) -> None:
        """A consistent copy of what is committed, as dest/memory.db."""
        dest.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest / DB_FILE)
        try:
            self.conn.backup(target)
        finally:
            target.close()

    def close(self) -> None:
        self.commit()
        self.conn.close()


# ---------------------------------------------------------------------------- JSON


class JsonBackend:
    """The files from before SQLite. Every save writes them in full."""

    name = "json"

    def __init__(self, home: Path) -> None:
        self.home = home
        self.vectors = JsonVectorIndex(home / VECTORS_FILE)

    def load(self) -> Rows:
        path = self.home / KNOWLEDGE_FILE
        if not path.exists():
            return {}
        data: Rows = json.loads(path.read_text(encoding="utf-8"))
        return data

    def save(self, store: KnowledgeStore, full: bool) -> None:
        payload = {
            "schema": KNOWLEDGE_SCHEMA,
            **{
                ("entities" if kind == "entity" else kind + "s"): [
                    o.model_dump(mode="json") for o in live.values()
                ]
                for kind, live in store.collections().items()
            },
        }
        atomic_write(self.home / KNOWLEDGE_FILE, json.dumps(payload, indent=1))
        self.vectors.save()

    def log_deletion(self, batch: dict[str, Any]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with (self.home / DELETED_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(batch) + "\n")

    def deletions(self) -> list[dict[str, Any]]:
        path = self.home / DELETED_FILE
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def drop_deletion(self, batch_id: str) -> None:
        rest = [b for b in self.deletions() if b["batch"] != batch_id]
        atomic_write(self.home / DELETED_FILE, "".join(json.dumps(b) + "\n" for b in rest))

    def summaries(self) -> dict[str, Any]:
        try:
            data = json.loads((self.home / SUMMARY_FILE).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def set_summaries(self, data: dict[str, Any]) -> None:
        atomic_write(self.home / SUMMARY_FILE, json.dumps(data, indent=1))

    def copy_to(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for name in JSON_FILES:
            if (self.home / name).exists():
                shutil.copy2(self.home / name, dest / name)

    def close(self) -> None:
        pass


def open_backend(home: Path) -> Backend:
    """memory.db when there is one; the JSON files when only they exist (a memory not
    migrated yet); for a new memory, SQLite when this Python can use it."""
    if (home / DB_FILE).exists():
        return SqliteBackend(home / DB_FILE)
    if (home / KNOWLEDGE_FILE).exists() or sqlite_problem() is not None:
        return JsonBackend(home)
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:  # nowhere to write: a store kept in memory (JSON writes only on save)
        return JsonBackend(home)
    return SqliteBackend(home / DB_FILE)


def differences(a: KnowledgeStore, b: KnowledgeStore, queries: int = 50) -> list[str]:
    """How two stores differ: every item, entity, edge, graduation, vector, deletion and
    summary, and the nearest notes for `queries` items' vectors. Empty when they hold the
    same memory. What `gm migrate` checks before it switches, and the simulator after
    every scenario."""
    out = []
    for kind, mine in a.collections().items():
        theirs = b.collections()[kind]
        if set(mine) != set(theirs):
            out.append(f"{kind}: {len(set(mine) ^ set(theirs))} ids differ")
            continue
        bad = [
            k for k in mine if mine[k].model_dump(mode="json") != theirs[k].model_dump(mode="json")
        ]
        if bad:
            out.append(f"{kind}: {len(bad)} differ, e.g. {bad[0]}")
    ids = [*a.items, *a.entities]
    if sum(a.vectors.get(i) is not None for i in ids) != sum(
        b.vectors.get(i) is not None for i in ids
    ):
        out.append("vectors: not the same ids")
    elif any(a.vectors.get(i) != b.vectors.get(i) for i in ids):
        out.append("vectors: some differ")
    if a.deleted() != b.deleted():
        out.append("deletions differ")
    if a.summaries() != b.summaries():
        out.append("summaries differ")
    for i in list(a.items)[:queries]:
        v = a.vectors.get(i)
        if v is None:
            continue
        near_a = [n for n, _ in a.vectors.query(v, 12, where={"kind": "item"})]
        near_b = [n for n, _ in b.vectors.query(v, 12, where={"kind": "item"})]
        if near_a != near_b:
            out.append(f"the nearest notes to {i} differ")
            break
    return out


def copy_memory(src: KnowledgeStore, dst: KnowledgeStore) -> dict[str, int]:
    """Copy all of `src` into the empty store `dst` and save it in full: items, entities,
    edges, graduations, vectors exactly as stored, deletions and summaries. A vector
    whose id is neither an item nor an entity (left by an old bug) is not carried over."""
    for kind, live in src.collections().items():
        dst.collections()[kind].update(live)
    vectors = 0
    for kind, ids in (("item", src.items), ("entity", src.entities)):
        for i in ids:
            v = src.vectors.get(i)
            if v is not None:
                dst.vectors.upsert(i, v, src.vector_meta(kind, i), unit=True)
                vectors += 1
    for batch in src.deleted():
        dst.backend.log_deletion(batch)
    summaries = src.summaries()
    if summaries:
        dst.backend.set_summaries(summaries)
    dst.save(full=True)
    return {
        "items": len(src.items),
        "entities": len(src.entities),
        "edges": len(src.edges),
        "segments": len(src.segments),
        "vectors": vectors,
        "deletions": len(src.deleted()),
        "summaries": len(summaries),
    }


def export_json(store: KnowledgeStore, dest: Path) -> dict[str, int]:
    """Write a memory out in the JSON format (`gm export`): the files a version from
    before SQLite reads, and readable in any editor."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in JSON_FILES:
        (dest / name).unlink(missing_ok=True)
    from .knowledge import KnowledgeStore

    out = KnowledgeStore(dest, backend=JsonBackend(dest))
    counts = copy_memory(store, out)
    return counts
