"""Vectors and files: the embedder type, a small cosine index, atomic writes.

Only knowledge items and entities have vectors; one JSON file holds them all.
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import os
import struct
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
        self, node_id: str, vector: Sequence[float], metadata: Metadata | None = None
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
    ) -> list[tuple[str, float]]: ...
    def save(self) -> None: ...


def _unit(v: Sequence[float]) -> list[float]:
    vec = [float(x) for x in v]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ValueError("zero vector cannot be indexed")
    return [x / norm for x in vec]


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


def _encode(v: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode("ascii")


def _decode(v: str | Sequence[float]) -> list[float]:
    if isinstance(v, str):
        raw = base64.b64decode(v)
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    return [float(x) for x in v]


class JsonVectorIndex:
    """Exact cosine index in one JSON file, searched as one matrix multiplication:
    under a millisecond for tens of thousands of nodes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._dim: int | None = None
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, str]] = {}
        self._matrix: tuple[list[str], Any] | None = None  # see _table()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._dim = data.get("dim")
            self._vectors = {k: _decode(v) for k, v in data["vectors"].items()}
            self._meta = {k: dict(v) for k, v in data.get("meta", {}).items()}

    def upsert(
        self, node_id: str, vector: Sequence[float], metadata: Metadata | None = None
    ) -> None:
        v = _unit(vector)
        if self._dim is None:
            self._dim = len(v)
        elif len(v) != self._dim:
            raise ValueError(f"vector dim {len(v)} != index dim {self._dim}")
        self._vectors[node_id] = v
        self._meta[node_id] = dict(metadata or {})
        self._matrix = None

    def get(self, node_id: str) -> list[float] | None:
        return self._vectors.get(node_id)

    def remove(self, node_id: str) -> None:
        self._vectors.pop(node_id, None)
        self._meta.pop(node_id, None)
        self._matrix = None

    def query(
        self,
        vector: Sequence[float],
        k: int,
        exclude: frozenset[str] = frozenset(),
        where: Metadata | None = None,
        where_not: Metadata | None = None,
        among: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._vectors or k <= 0:
            return []
        ids, matrix = self._table()
        scores = matrix @ np.asarray(_unit(vector), dtype=np.float64)
        keep = [
            n
            for n, nid in enumerate(ids)
            if nid not in exclude
            and (among is None or nid in among)
            and _meta_matches(self._meta.get(nid, {}), where, where_not)
        ]
        if not keep:
            return []
        rows = np.asarray(keep)
        kept = scores[rows]
        top = np.argsort(-kept, kind="stable")[:k]
        return [(ids[int(rows[t])], float(kept[t])) for t in top]

    def _table(self) -> tuple[list[str], Any]:
        """Every vector as one row of a matrix, so a query is one multiplication.
        Built on the first query after a change."""
        if self._matrix is None:
            ids = list(self._vectors)
            self._matrix = (ids, np.asarray([self._vectors[i] for i in ids], dtype=np.float64))
        return self._matrix

    def save(self) -> None:
        payload = {
            "dim": self._dim,
            "encoding": ENCODING,
            "vectors": {k: _encode(v) for k, v in self._vectors.items()},
            "meta": self._meta,
        }
        atomic_write(self.path, json.dumps(payload))


def chroma_default_embedder() -> Embedder:
    """Chroma's built-in local model (all-MiniLM-L6-v2, ONNX, 384 dims). Downloads ~80 MB
    to ~/.cache/chroma on first use, then runs offline on CPU (~40 ms per text)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()

    def embed(text: str) -> list[float]:
        return [float(x) for x in ef([text])[0]]

    return embed


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
