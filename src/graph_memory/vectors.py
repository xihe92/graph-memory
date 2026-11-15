"""Vectors and files: the embedder type, a small cosine index, atomic writes.

Only knowledge items and entities have vectors; one JSON file holds them all.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

Embedder = Callable[[str], Sequence[float]]
Metadata = Mapping[str, str]


@runtime_checkable
class VectorIndex(Protocol):
    """Similarity search backend. Vectors are stored unit-normalised, so the dot product
    of two `get()` results is their cosine similarity. `where` keeps only entries whose
    metadata equals every given value; `where_not` drops entries matching any of them."""

    def upsert(
        self,
        node_id: str,
        vector: Sequence[float],
        metadata: Metadata | None = None,
        unit: bool = False,
    ) -> None: ...
    def get(self, node_id: str) -> list[float] | None: ...
    def remove(self, node_id: str) -> None: ...
    def query(
        self,
        vector: Sequence[float],
        k: int,
        exclude: frozenset[str] = frozenset(),
        where: Metadata | None = None,
        where_not: Metadata | None = None,
        among: set[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]: ...
    def set_meta(self, meta: Mapping[str, tuple[str, bool]]) -> None: ...
    def save(self) -> None: ...


Vec32 = np.ndarray[Any, np.dtype[np.float32]]


def _unit(v: Sequence[float] | Vec32) -> Vec32:
    vec = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("zero vector cannot be indexed")
    out: Vec32 = vec / np.float32(norm)
    return out


def _meta_matches(meta: Metadata, where: Metadata | None, where_not: Metadata | None) -> bool:
    if where and any(meta.get(k) != v for k, v in where.items()):
        return False
    if where_not and any(meta.get(k) == v for k, v in where_not.items()):
        return False
    return True


# Vectors are written as base64 of little-endian float32: the same precision the
# embedder produces, at about a third of the size of decimal text. Files written as
# lists of numbers still load.
ENCODING = "f32-base64"


def _encode(v: Vec32) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode("ascii")


def _decode(v: str | Sequence[float]) -> Vec32:
    if isinstance(v, str):
        out: Vec32 = np.frombuffer(base64.b64decode(v), dtype="<f4").astype(np.float32)
        return out
    return np.asarray([float(x) for x in v], dtype=np.float32)


class JsonVectorIndex:
    """Exact cosine index in one JSON file. Every vector is one row of a single float32
    array (4 bytes a number, no second copy), so a query is one multiplication over the
    rows it may consider."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._dim: int | None = None
        self._ids: list[str] = []  # row -> id
        self._row: dict[str, int] = {}  # id -> row
        self._mat: Vec32 = np.zeros((0, 0), dtype=np.float32)  # rows past len(_ids) unused
        self._meta: dict[str, dict[str, str]] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            for k, v in data["vectors"].items():  # stored at unit length already
                self._put(k, _decode(v), meta.get(k))

    def __len__(self) -> int:
        return len(self._ids)

    def upsert(
        self,
        node_id: str,
        vector: Sequence[float] | Vec32,
        metadata: Metadata | None = None,
        unit: bool = False,
    ) -> None:
        """`unit`: the vector is already scaled to length 1 (one read from storage): kept
        exactly as given."""
        v = np.asarray(vector, dtype=np.float32) if unit else _unit(vector)
        self._put(node_id, v, metadata)

    def _put(self, node_id: str, v: Vec32, metadata: Metadata | None) -> None:
        if self._dim is None:
            self._dim = len(v)
            self._mat = np.zeros((16, self._dim), dtype=np.float32)
        elif len(v) != self._dim:
            raise ValueError(f"vector dim {len(v)} != index dim {self._dim}")
        row = self._row.get(node_id)
        if row is None:
            row = len(self._ids)
            if row == len(self._mat):  # full: double the room
                grown = np.zeros((2 * len(self._mat), self._dim), dtype=np.float32)
                grown[:row] = self._mat[:row]
                self._mat = grown
            self._ids.append(node_id)
            self._row[node_id] = row
        self._mat[row] = v
        # the kind only: a search here is limited by `among`, so a scope kept beside the
        # vector would only go stale in the file
        kind = (metadata or {}).get("kind")
        self._meta[node_id] = {"kind": kind} if kind is not None else {}

    def get(self, node_id: str) -> list[float] | None:
        row = self._row.get(node_id)
        return None if row is None else [float(x) for x in self._mat[row]]

    def remove(self, node_id: str) -> None:
        row = self._row.pop(node_id, None)
        self._meta.pop(node_id, None)
        if row is None:
            return
        last = len(self._ids) - 1
        if row != last:  # the last row fills the hole
            moved = self._ids[last]
            self._mat[row] = self._mat[last]
            self._ids[row] = moved
            self._row[moved] = row
        self._ids.pop()

    def query(
        self,
        vector: Sequence[float],
        k: int,
        exclude: frozenset[str] = frozenset(),
        where: Metadata | None = None,
        where_not: Metadata | None = None,
        among: set[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        ids = self._ids
        # the rows to consider: only the given ids when a set is given, so a search
        # limited to a scope does not walk every vector
        pool = (
            sorted(self._row[i] for i in among if i in self._row)
            if among is not None
            else range(len(ids))
        )
        keep = [
            n
            for n in pool
            if ids[n] not in exclude and _meta_matches(self._meta.get(ids[n], {}), where, where_not)
        ]
        if not keep:
            return []
        rows = np.asarray(keep)
        kept = self._mat[rows] @ _unit(vector)
        top = np.argsort(-kept, kind="stable")[:k]
        return [(ids[int(rows[t])], float(kept[t])) for t in top]

    def set_meta(self, meta: Mapping[str, tuple[str, bool]]) -> None:
        pass  # this index limits a search by `among`, which says the same

    def save(self) -> None:
        payload = {
            "dim": self._dim,
            "encoding": ENCODING,
            "vectors": {i: _encode(self._mat[n]) for n, i in enumerate(self._ids)},
            "meta": self._meta,
        }
        atomic_write(self.path, json.dumps(payload))


KNN_MAX = 4096  # the most rows sqlite-vec returns from one nearest-neighbour query


class SqliteVectorIndex:
    """Vectors in memory.db, searched in place by sqlite-vec (an exact cosine scan, done
    in C by SQLite): nothing is held in Python's memory. Writes go into the backend's
    open transaction, so they reach disk with the save that commits the notes."""

    TABLE = "vec"

    def __init__(self, conn: Any, begin: Callable[[], None]) -> None:
        self.conn = conn
        self.begin = begin
        row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        self._dim: int | None = int(row[0]) if row else None
        self._counts: dict[str, int] | None = None  # rows per kind, see _count()

    def __len__(self) -> int:
        return sum(self._count().values())

    def _ensure(self, dim: int) -> None:
        if self._dim is None:
            self.begin()
            # beside each vector: what it is, and for an item its scope and whether it can
            # be recalled now, so a search for a session filters inside the scan
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.TABLE} USING vec0("
                f"id TEXT PRIMARY KEY, kind TEXT, scope TEXT, live INTEGER,"
                f" embedding float[{dim}] distance_metric=cosine)"
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('dim', ?)", (str(dim),))
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(f"vector dim {dim} != index dim {self._dim}")

    def _count(self) -> dict[str, int]:
        if self._counts is None:
            self._counts = (
                dict(self.conn.execute(f"SELECT kind, count(*) FROM {self.TABLE} GROUP BY kind"))
                if self._dim is not None
                else {}
            )
        return self._counts

    def upsert(
        self,
        node_id: str,
        vector: Sequence[float] | Vec32,
        metadata: Metadata | None = None,
        unit: bool = False,
    ) -> None:
        v = np.asarray(vector, dtype=np.float32) if unit else _unit(vector)
        self._ensure(len(v))
        self.begin()
        self.conn.execute(f"DELETE FROM {self.TABLE} WHERE id = ?", (node_id,))
        meta = metadata or {}
        self.conn.execute(
            f"INSERT INTO {self.TABLE}(id, kind, scope, live, embedding) VALUES (?, ?, ?, ?, ?)",
            (
                node_id,
                meta.get("kind", ""),
                meta.get("scope", ""),
                int(str(meta.get("live", "1")) == "1"),
                np.asarray(v, dtype="<f4").tobytes(),
            ),
        )
        self._counts = None

    def set_meta(self, meta: Mapping[str, tuple[str, bool]]) -> None:
        """File items' scope and recallability (id -> (scope, live)) beside their vectors,
        writing only rows that differ. A few are looked up by id; for many, one scan of
        the items' rows is quicker."""
        if self._dim is None or not meta:
            return
        if len(meta) <= 64:
            rows = [
                (i, *r)
                for i in meta
                for r in self.conn.execute(
                    f"SELECT scope, live FROM {self.TABLE} WHERE id = ?", (i,)
                ).fetchall()
            ]
        else:
            rows = self.conn.execute(
                f"SELECT id, scope, live FROM {self.TABLE} WHERE kind = 'item'"
            ).fetchall()
        stale = []
        for i, scope, live in rows:
            want = meta.get(i)
            if want is not None and (scope, live) != (want[0], int(want[1])):
                stale.append((want[0], int(want[1]), i))
        if stale:
            self.begin()
            self.conn.executemany(
                f"UPDATE {self.TABLE} SET scope = ?, live = ? WHERE id = ?", stale
            )

    def get(self, node_id: str) -> list[float] | None:
        if self._dim is None:
            return None
        row = self.conn.execute(
            f"SELECT embedding FROM {self.TABLE} WHERE id = ?", (node_id,)
        ).fetchone()
        return None if row is None else [float(x) for x in np.frombuffer(row[0], dtype="<f4")]

    def remove(self, node_id: str) -> None:
        if self._dim is None:
            return
        self.begin()
        self.conn.execute(f"DELETE FROM {self.TABLE} WHERE id = ?", (node_id,))
        self._counts = None

    def query(
        self,
        vector: Sequence[float],
        k: int,
        exclude: frozenset[str] = frozenset(),
        where: Metadata | None = None,
        where_not: Metadata | None = None,
        among: set[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """The k nearest, exactly. `scopes` (a recall's) filters inside the scan: items in
        those scopes that can be recalled now. `among` then keeps only the given ids;
        when that leaves fewer than k, it asks again for more (sqlite-vec returns at most
        4096 rows a query)."""
        if self._dim is None or k <= 0 or (among is not None and not among):
            return []
        if where_not or set(where or {}) - {"kind"}:
            raise ValueError("SqliteVectorIndex filters on kind only")
        kind = (where or {}).get("kind")
        counts = self._count()
        n = counts.get(kind, 0) if kind is not None else sum(counts.values())
        if not n:
            return []
        q = np.asarray(_unit(vector), dtype="<f4").tobytes()
        sql = f"SELECT id, distance FROM {self.TABLE} WHERE embedding MATCH ? AND k = ?"
        extra: list[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            extra.append(kind)
        if scopes is not None:
            sql += " AND live = 1 AND scope IN (SELECT value FROM json_each(?))"
            extra.append(json.dumps(list(scopes)))
            fetch = k + len(exclude)
        else:
            fetch = k + len(exclude) + (max(0, n - len(among)) if among is not None else 0)
        fetch = min(n, fetch, KNN_MAX)
        while True:
            args: list[Any] = [q, fetch, *extra]
            out = []
            for nid, dist in self.conn.execute(sql, args):
                if nid in exclude or (among is not None and nid not in among):
                    continue
                out.append((nid, 1.0 - float(dist)))
                if len(out) == k:
                    return out
            if fetch >= min(n, KNN_MAX):  # read all it can: fewer than k qualify
                return out
            fetch = min(n, KNN_MAX, fetch * 4)  # some rows were left out: ask for more

    def save(self) -> None:
        pass  # the backend commits


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
