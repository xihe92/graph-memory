"""Where knowledge is kept on disk: memory.db, one SQLite file.

    item, entity, edge, segment  one row each, the object as JSON, with a few columns
                                 beside it for SQL to filter on
    vec                          every vector, searched in place by sqlite-vec
    word, doc                    recall's word index: for each note recall can show, how
                                 often each of its words occurs, and its length
    deleted, summary             deletions that can be undone; project and topic summaries
    active_day                   the dates you worked, in all and per project, which
                                 notes age by (see active_days.py)

The store (knowledge.py) keeps only small fields in memory and reads the rest from here
by id. Its changes go into an open transaction, which its save() commits, so notes, their
vectors and their words reach disk together or not at all.

The JSON files from before SQLite are read by `gm migrate` and written by `gm export`,
as an Image: the whole memory as plain data.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .files import atomic_write
from .vectors import JsonVectorIndex, SqliteVectorIndex

DB_FILE = "memory.db"
KNOWLEDGE_FILE = "knowledge.json"
VECTORS_FILE = "knowledge_vectors.json"
DELETED_FILE = "knowledge_deleted.jsonl"
SUMMARY_FILE = "summaries.json"
JSON_FILES = (KNOWLEDGE_FILE, VECTORS_FILE, DELETED_FILE, SUMMARY_FILE)
KNOWLEDGE_SCHEMA = 1
DB_SCHEMA = 3  # 2: the word index; 3: active days


class StorageError(RuntimeError):
    pass


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


# kind: (table, key columns, the extra columns written beside the JSON body)
TABLES = {
    "item": ("item", ("id",), ("type", "scope", "valid")),
    "entity": ("entity", ("id",), ("name", "kind")),
    "edge": ("edge", ("source", "target", "relation"), ()),
    "segment": ("segment", ("id",), ("session",)),
}

_WORD_TABLES = (
    # for each note recall can show: each of its words, how often, with the note's length
    # and scope beside it, so one query scores the notes of a recall's scopes
    "CREATE TABLE IF NOT EXISTS word(term TEXT, item TEXT, tf INTEGER, dl INTEGER,"
    " scope TEXT, PRIMARY KEY (term, item)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS word_item ON word(item)",
    "CREATE TABLE IF NOT EXISTS doc(item TEXT PRIMARY KEY, len INTEGER)",
)

_DAY_TABLE = (
    "CREATE TABLE IF NOT EXISTS active_day(date TEXT, scope TEXT, PRIMARY KEY (date, scope))"
    " WITHOUT ROWID"
)

_CHUNK = 500  # ids per IN (...) query


def _chunks(ids: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for n in range(0, len(ids), _CHUNK):
        yield ids[n : n + _CHUNK]


class SqliteBackend:
    """memory.db. Writes go into an open transaction (vectors as soon as they change,
    rows when the store flushes them) that commit() ends."""

    name = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = connect(path)
        c = self.conn
        # set up a new file, or bring an older one up to date; a current one is only
        # read here, so opening it never waits for (or blocks) the daemon's writes
        new = c.execute("SELECT 1 FROM sqlite_master WHERE name = 'item'").fetchone() is None
        old = not new and any(
            c.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (t,)).fetchone() is None
            for t in ("doc", "active_day")
        )
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
        elif old:
            c.execute("BEGIN IMMEDIATE")
        if new or old:
            c.execute("CREATE INDEX IF NOT EXISTS segment_session ON segment(session)")
            for sql in (*_WORD_TABLES, _DAY_TABLE):
                c.execute(sql)
            c.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", (str(DB_SCHEMA),))
            if new:  # empty: built and filled
                c.execute("INSERT OR REPLACE INTO meta VALUES ('words', '1')")
                c.execute("INSERT OR REPLACE INTO meta VALUES ('days', '1')")
            c.execute("COMMIT")
        # the word index is built by the store (it splits the words) the first time it
        # opens a memory, and marked built in the same transaction
        row = c.execute("SELECT value FROM meta WHERE key = 'words'").fetchone()
        self.needs_words = row is None
        # likewise the active days of a memory from before they were kept
        row = c.execute("SELECT value FROM meta WHERE key = 'days'").fetchone()
        self.needs_days = row is None
        self.vectors = SqliteVectorIndex(c, self.begin)

    def begin(self) -> None:
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN")

    def commit(self) -> None:
        if self.conn.in_transaction:
            self.conn.execute("COMMIT")

    def rollback(self) -> None:
        if self.conn.in_transaction:
            self.conn.execute("ROLLBACK")

    # ------------------------------------------------------------------ reading

    def item_heads(self) -> Iterator[tuple[Any, ...]]:
        """For every item, in the order they were added: the fields the store keeps in
        memory, read out of the JSON without building the objects."""
        return self.conn.execute(
            "SELECT id, type, scope, valid, json_extract(body, '$.area'),"
            " json_extract(body, '$.strength'), json_extract(body, '$.last_seen'),"
            " json_extract(body, '$.last_used'), json_extract(body, '$.volatility'),"
            " json_extract(body, '$.checked_at'), json_extract(body, '$.created_at'),"
            " json_extract(body, '$.entities'), json_extract(body, '$.statement'),"
            " json_extract(body, '$.sources'),"
            " length(json_extract(body, '$.statement')) + length(json_extract(body, '$.why'))"
            " FROM item ORDER BY rowid"
        )

    def entity_heads(self) -> Iterator[tuple[Any, ...]]:
        return self.conn.execute(
            "SELECT id, name, kind, json_extract(body, '$.aliases'),"
            " json_extract(body, '$.mentions') FROM entity ORDER BY rowid"
        )

    def edge_weights(self) -> Iterator[tuple[str, str, str, float]]:
        return self.conn.execute(
            "SELECT source, target, relation, json_extract(body, '$.weight') FROM edge"
            " ORDER BY rowid"
        )

    def bodies(self, kind: str, keys: Sequence[Any]) -> dict[Any, str]:
        """The JSON of the given rows of one kind, by key (an edge's key is its three
        columns); missing ones are left out."""
        table, cols, _ = TABLES[kind]
        out: dict[Any, str] = {}
        if not keys:
            return out
        if len(cols) == 1:
            for part in _chunks(list(keys)):
                marks = ",".join("?" * len(part))
                sql = f"SELECT {cols[0]}, body FROM {table} WHERE {cols[0]} IN ({marks})"  # noqa: S608
                out.update(self.conn.execute(sql, list(part)))
            return out
        sql = f"SELECT body FROM {table} WHERE source = ? AND target = ? AND relation = ?"  # noqa: S608
        for key in keys:
            row = self.conn.execute(sql, key).fetchone()
            if row is not None:
                out[key] = row[0]
        return out

    def scan(self, kind: str, where: str = "", args: Sequence[Any] = ()) -> Iterator[str]:
        """The JSON of every row of a kind matching `where`, in the order added."""
        table = TABLES[kind][0]
        sql = f"SELECT body FROM {table} {'WHERE ' + where if where else ''} ORDER BY rowid"  # noqa: S608
        return (b for (b,) in self.conn.execute(sql, list(args)))

    def keys(self, kind: str, where: str = "", args: Sequence[Any] = ()) -> list[Any]:
        table, cols, _ = TABLES[kind]
        sql = (
            f"SELECT {', '.join(cols)} FROM {table} {'WHERE ' + where if where else ''}"  # noqa: S608
            " ORDER BY rowid"
        )
        rows = self.conn.execute(sql, list(args))
        return [r[0] if len(cols) == 1 else tuple(r) for r in rows]

    def texts(self) -> Iterator[tuple[str, str, str, str]]:
        """Every item's id, statement, reason and entity ids (JSON), in the order added."""
        return self.conn.execute(
            "SELECT id, json_extract(body, '$.statement'), json_extract(body, '$.why'),"
            " json_extract(body, '$.entities') FROM item ORDER BY rowid"
        )

    def fields(self, kind: str, names: Sequence[str]) -> Iterator[tuple[Any, ...]]:
        """For every row of a kind, in the order added: its key columns, then the named
        fields read out of its JSON (a list or object comes back as JSON text)."""
        table, cols, _ = TABLES[kind]
        picks = ", ".join(f"json_extract(body, '$.{n}')" for n in names)
        return self.conn.execute(
            f"SELECT {', '.join(cols)}, {picks} FROM {table} ORDER BY rowid"  # noqa: S608
        )

    def segment_sessions(self) -> set[str]:
        return {s for (s,) in self.conn.execute("SELECT DISTINCT session FROM segment")}

    def count(self, kind: str) -> int:
        row = self.conn.execute(f"SELECT count(*) FROM {TABLES[kind][0]}").fetchone()  # noqa: S608
        return int(row[0])

    # ------------------------------------------------------------------ writing

    def write(self, kind: str, obj: Any) -> None:
        """Insert or update one row; an update keeps the row's place in the order."""
        table, keys, extra = TABLES[kind]
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
        cols = (*keys, *extra, "body")
        marks = ", ".join("?" * len(vals))
        sets = ", ".join(f"{c} = excluded.{c}" for c in (*extra, "body"))
        self.begin()
        self.conn.execute(
            f"INSERT INTO {table}({', '.join(cols)}) VALUES ({marks})"  # noqa: S608
            f" ON CONFLICT({', '.join(keys)}) DO UPDATE SET {sets}",
            vals,
        )

    def remove(self, kind: str, key: Any) -> None:
        table, keys, _ = TABLES[kind]
        where = " AND ".join(f"{k} = ?" for k in keys)
        args = key if isinstance(key, tuple) else (key,)
        self.begin()
        self.conn.execute(f"DELETE FROM {table} WHERE {where}", args)  # noqa: S608

    # ------------------------------------------------------------------ words

    def doc_stats(self) -> tuple[int, int]:
        """How many notes the word index holds, and their total length."""
        n, total = self.conn.execute("SELECT count(*), coalesce(sum(len), 0) FROM doc").fetchone()
        return int(n), int(total)

    def set_words(
        self, item: str, tf: Mapping[str, int] | None, scope: str = ""
    ) -> tuple[int | None, int | None]:
        """Replace an item's words (None: take it out of the index); nothing is written
        when they are the same. Returns its length before and after, None when it was not,
        or is no longer, in the index."""
        row = self.conn.execute("SELECT len FROM doc WHERE item = ?", (item,)).fetchone()
        before = int(row[0]) if row else None
        if row and tf is not None:
            have = {
                t: (f, sc)
                for t, f, sc in self.conn.execute(
                    "SELECT term, tf, scope FROM word WHERE item = ?", (item,)
                )
            }
            if have == {t: (f, scope) for t, f in tf.items()}:
                return before, before
        if row is None and tf is None:
            return None, None
        self.begin()
        if row:
            self.conn.execute("DELETE FROM word WHERE item = ?", (item,))
            self.conn.execute("DELETE FROM doc WHERE item = ?", (item,))
        if tf is None:
            return before, None
        dl = sum(tf.values())
        self.conn.executemany(
            "INSERT INTO word VALUES (?, ?, ?, ?, ?)",
            [(t, item, f, dl, scope) for t, f in tf.items()],
        )
        self.conn.execute("INSERT INTO doc VALUES (?, ?)", (item, dl))
        return before, dl

    def words_built(self) -> None:
        """Mark the word index built (committed with the store's save)."""
        self.begin()
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('words', '1')")
        self.needs_words = False

    def word_df(self, terms: Iterable[str]) -> dict[str, int]:
        """How many notes in the index hold each word (words in none are left out)."""
        want = list(dict.fromkeys(terms))
        out: dict[str, int] = {}
        for part in _chunks(want):
            marks = ",".join("?" * len(part))
            out.update(
                self.conn.execute(
                    f"SELECT term, count(*) FROM word WHERE term IN ({marks}) GROUP BY term",  # noqa: S608
                    list(part),
                )
            )
        return out

    def word_scores(
        self,
        idf: Mapping[str, float],
        avg_len: float,
        k: int,
        scopes: Sequence[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> list[tuple[str, float, int]]:
        """BM25 (k1 1.2, b 0.75) of the notes holding any of the words, with the given
        idf per word: the best k as (item, score, how many of the words it holds), best
        first, ties by id."""
        if not idf or k <= 0:
            return []
        terms = list(idf.items())
        values = ",".join("(?, ?)" for _ in terms)
        args: list[Any] = [x for t in terms for x in t]
        sql = (
            f"WITH q(term, idf) AS (VALUES {values})"  # noqa: S608
            " SELECT w.item, SUM(q.idf * w.tf * 2.2 / (w.tf + 1.2 * (0.25 + 0.75 * w.dl / ?))),"
            " count(*) FROM word w JOIN q ON w.term = q.term"
        )
        args.append(avg_len)
        where = []
        if scopes is not None:
            if not scopes:
                return []
            where.append(f"w.scope IN ({','.join('?' * len(scopes))})")
            args.extend(scopes)
        skip = list(exclude)
        if skip:
            where.append(f"w.item NOT IN ({','.join('?' * len(skip))})")
            args.extend(skip)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY w.item ORDER BY 2 DESC, w.item LIMIT ?"
        args.append(k)
        return [(i, float(s), int(n)) for i, s, n in self.conn.execute(sql, args)]

    def words_of(self, item: str) -> dict[str, int]:
        return dict(self.conn.execute("SELECT term, tf FROM word WHERE item = ?", (item,)))

    def word_rows(self) -> Iterator[tuple[str, str, int, int, str]]:
        return self.conn.execute("SELECT term, item, tf, dl, scope FROM word ORDER BY item, term")

    # ------------------------------------------------------------------ the rest

    def log_deletion(self, batch: dict[str, Any]) -> None:
        self.begin()  # committed with the save that follows the deletion
        self.conn.execute(
            "INSERT OR REPLACE INTO deleted VALUES (?, ?, ?)",
            (batch["batch"], batch["at"], json.dumps(batch)),
        )

    def deletions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT body FROM deleted ORDER BY at, rowid")
        return [json.loads(b) for (b,) in rows]

    def deletion(self, batch_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT body FROM deleted WHERE batch = ?", (batch_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def drop_deletion(self, batch_id: str) -> None:
        self.begin()
        self.conn.execute("DELETE FROM deleted WHERE batch = ?", (batch_id,))

    def active_days(self) -> list[tuple[str, str]]:
        return [tuple(r) for r in self.conn.execute("SELECT date, scope FROM active_day")]

    def add_days(self, rows: Iterable[tuple[str, str]], *, filled: bool = False) -> None:
        """Add active days (date, scope); `filled`: the one-time fill is done. Committed
        with the store's save."""
        self.begin()
        self.conn.executemany("INSERT OR IGNORE INTO active_day VALUES (?, ?)", rows)
        if filled:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('days', '1')")
            self.needs_days = False

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


def open_backend(home: Path) -> SqliteBackend:
    """memory.db in the memory folder, made when there is none. A memory still in the
    JSON files from before SQLite is moved by `gm migrate` first."""
    if not (home / DB_FILE).exists() and (home / KNOWLEDGE_FILE).exists():
        raise StorageError(
            f"memory in {home} is in the JSON files from before SQLite; `gm stop`, then "
            "`gm migrate`, moves it to memory.db"
        )
    home.mkdir(parents=True, exist_ok=True)
    return SqliteBackend(home / DB_FILE)


# ---------------------------------------------------------------------------- images


@dataclass
class Image:
    """A whole memory as plain data: what `gm migrate` reads from the JSON files and
    `gm export` writes to them. Rows are the objects as JSON, in the order added;
    vectors are unit length."""

    items: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)  # vector id -> "item" or "entity"
    deletions: list[dict[str, Any]] = field(default_factory=list)
    summaries: dict[str, Any] = field(default_factory=dict)
    # (date, scope), oldest first; None for JSON files from before they were kept, whose
    # store fills them in from the notes when it first opens them
    active_days: list[tuple[str, str]] | None = None

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "item": self.items,
            "entity": self.entities,
            "edge": self.edges,
            "segment": self.segments,
        }

    def counts(self) -> dict[str, int]:
        return {
            "items": len(self.items),
            "entities": len(self.entities),
            "edges": len(self.edges),
            "segments": len(self.segments),
            "vectors": len(self.vectors),
            "deletions": len(self.deletions),
            "summaries": len(self.summaries),
            "active days": len(self.active_days or ()),
        }


def read_json(home: Path) -> Image:
    """The memory in the JSON files from before SQLite."""
    img = Image()
    path = home / KNOWLEDGE_FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        img.items = list(data.get("items") or [])
        img.entities = list(data.get("entities") or [])
        img.edges = list(data.get("edges") or [])
        img.segments = list(data.get("segments") or [])
        if "active_days" in data:
            img.active_days = sorted((str(d), str(s)) for d, s in data["active_days"])
    if (home / VECTORS_FILE).exists():
        ix = JsonVectorIndex(home / VECTORS_FILE)
        known = {"item": {i["item_id"] for i in img.items}}
        known["entity"] = {e["entity_id"] for e in img.entities}
        for kind, ids in known.items():
            for i in ids:
                v = ix.get(i)
                if v is not None:
                    img.vectors[i] = v
                    img.kinds[i] = kind
    dpath = home / DELETED_FILE
    if dpath.exists():
        for line in dpath.read_text(encoding="utf-8").splitlines():
            try:
                img.deletions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    try:
        summaries = json.loads((home / SUMMARY_FILE).read_text())
        img.summaries = summaries if isinstance(summaries, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return img


def write_json(img: Image, dest: Path) -> None:
    """The memory as the JSON files from before SQLite, in `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in JSON_FILES:
        (dest / name).unlink(missing_ok=True)
    payload = {
        "schema": KNOWLEDGE_SCHEMA,
        "entities": img.entities,
        "items": img.items,
        "edges": img.edges,
        "segments": img.segments,
    }
    if img.active_days is not None:
        payload["active_days"] = img.active_days
    atomic_write(dest / KNOWLEDGE_FILE, json.dumps(payload, indent=1))
    ix = JsonVectorIndex(dest / VECTORS_FILE)
    for i, v in img.vectors.items():
        ix.upsert(i, v, {"kind": img.kinds.get(i, "item")}, unit=True)
    ix.save()
    if img.deletions:
        (dest / DELETED_FILE).write_text(
            "".join(json.dumps(b) + "\n" for b in img.deletions), encoding="utf-8"
        )
    if img.summaries:
        atomic_write(dest / SUMMARY_FILE, json.dumps(img.summaries, indent=1))


def read_db(db: SqliteBackend) -> Image:
    """Everything in memory.db (what is committed, and what its connection has written)."""
    img = Image()
    for kind, rows in img.rows().items():
        rows.extend(json.loads(b) for b in db.scan(kind))
    for kind, rows in (("item", img.items), ("entity", img.entities)):
        key = "item_id" if kind == "item" else "entity_id"
        for r in rows:
            v = db.vectors.get(r[key])
            if v is not None:
                img.vectors[r[key]] = v
                img.kinds[r[key]] = kind
    img.deletions = db.deletions()
    img.summaries = db.summaries()
    if not db.needs_days:
        img.active_days = sorted(db.active_days())
    return img


def write_db(img: Image, db: SqliteBackend) -> None:
    """Put an image into an empty memory.db and commit it. The word index is left to the
    store that next opens it."""
    from .knowledge import Item

    models = _models()
    db.begin()
    try:
        for kind, rows in img.rows().items():
            for r in rows:
                db.write(kind, models[kind].model_validate(r))
        scope = {r["item_id"]: (r.get("scope") or "global", r) for r in img.items}
        for i, v in img.vectors.items():
            kind = img.kinds.get(i, "item")
            meta = {"kind": kind}
            if kind == "item" and i in scope:
                it = Item.model_validate(scope[i][1])
                meta |= {"scope": it.scope, "live": "1" if it.recallable else "0"}
            db.vectors.upsert(i, v, meta, unit=True)
        for batch in img.deletions:
            db.log_deletion(batch)
        if img.summaries:
            db.set_summaries(img.summaries)
        db.conn.execute("DELETE FROM active_day")
        db.conn.execute("DELETE FROM meta WHERE key = 'days'")
        if img.active_days is not None:
            db.add_days(img.active_days, filled=True)
        # the word index is built by the store that next opens it
        db.conn.execute("DELETE FROM doc")
        db.conn.execute("DELETE FROM word")
        db.conn.execute("DELETE FROM meta WHERE key = 'words'")
        db.commit()
    except BaseException:
        db.rollback()
        raise
    db.needs_words = True
    db.needs_days = img.active_days is None


def differences(a: Image, b: Image, queries: int = 50) -> list[str]:
    """How two memories differ: every item, entity, edge, graduation, vector, deletion
    and summary, and the nearest notes for `queries` items' vectors. Empty when they hold
    the same memory. What `gm migrate` checks before it switches."""
    out = []
    for kind, mine in a.rows().items():
        ra = {_key(kind, r): r for r in mine}
        rb = {_key(kind, r): r for r in b.rows()[kind]}
        if set(ra) != set(rb):
            out.append(f"{kind}: {len(set(ra) ^ set(rb))} ids differ")
            continue
        bad = [k for k in ra if _norm(kind, ra[k]) != _norm(kind, rb[k])]
        if bad:
            out.append(f"{kind}: {len(bad)} differ, e.g. {bad[0]}")
    if set(a.vectors) != set(b.vectors):
        out.append("vectors: not the same ids")
    elif any(not _close(a.vectors[i], b.vectors[i]) for i in a.vectors):
        out.append("vectors: some differ")
    if a.deletions != b.deletions:
        out.append("deletions differ")
    if a.summaries != b.summaries:
        out.append("summaries differ")
    if a.active_days != b.active_days:
        out.append("active days differ")
    if not out:
        ia, ib = _index(a), _index(b)
        for i in [r["item_id"] for r in a.items][:queries]:
            v = a.vectors.get(i)
            if v is None:
                continue
            near_a = [n for n, _ in ia.query(v, 12, where={"kind": "item"})]
            near_b = [n for n, _ in ib.query(v, 12, where={"kind": "item"})]
            if near_a != near_b:
                out.append(f"the nearest notes to {i} differ")
                break
    return out


def _key(kind: str, row: dict[str, Any]) -> Any:
    if kind == "edge":
        return (row["source"], row["target"], row["relation"])
    return row["item_id" if kind == "item" else f"{kind}_id"]


def _models() -> dict[str, type[BaseModel]]:
    from .knowledge import Entity, Item, ItemEdge, Segment

    return {"item": Item, "entity": Entity, "edge": ItemEdge, "segment": Segment}


def _norm(kind: str, row: dict[str, Any]) -> Any:
    """A row as its model writes it, so the same object written by two versions compares
    equal (defaults filled in, dates in one format)."""
    return _models()[kind].model_validate(row).model_dump(mode="json")


def _close(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(
        math.isclose(x, y, abs_tol=1e-7) for x, y in zip(a, b, strict=True)
    )


def _index(img: Image) -> JsonVectorIndex:
    ix = JsonVectorIndex(Path("/nonexistent/vectors.json"))  # in memory only, never saved
    for i, v in img.vectors.items():
        ix.upsert(i, v, {"kind": img.kinds.get(i, "item")}, unit=True)
    return ix
