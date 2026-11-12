"""The session buffer: a session's exchanges, kept only until they graduate.

    <home>/sessions/<session>/chain.jsonl     one record per line, oldest first

An *exchange* record is one prompt and its reply, appended when the reply arrives. A
*summary* record replaces the exchanges a graduation covered: Claude's own summary after
a compaction, or the summary of the session so far that the graduation wrote. The next
graduation of the session reads the latest summary as context, and folds into it: a
session keeps one summary of itself, plus one per compaction. When the session
ends and its last exchanges graduate, the folder is deleted: only knowledge is kept for
good.

Appends are one line each. Every other change rewrites the file atomically (temporary
file, fsync, rename), so a crash leaves the old file or the new one, never half of each.
The daemon is the only writer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote, unquote

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from . import clock
from .anchors import CodeAnchor, SourceRef
from .vectors import atomic_write

log = logging.getLogger(__name__)

SESSIONS_DIR = "sessions"
CHAIN_FILE = "chain.jsonl"


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Exchange(BaseModel):
    kind: Literal["exchange"] = "exchange"
    id: str = Field(default_factory=_id)
    ts: datetime = Field(default_factory=clock.now)
    prompt: str
    reply: str
    files: list[CodeAnchor] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    web: list[str] = Field(default_factory=list)  # pages read and searches run
    project: str | None = None  # "repo:/path", from the files it touched or the session
    source: SourceRef | None = None  # where the turn lives in the agent's transcript


class Summary(BaseModel):
    kind: Literal["summary"] = "summary"
    id: str = Field(default_factory=_id)
    ts: datetime = Field(default_factory=clock.now)  # when it replaced the exchanges
    trigger: str  # compact, idle, ...
    start: datetime | None = None  # first and last exchange it replaced
    end: datetime | None = None
    exchanges: int = 0
    segment: str | None = None  # the (latest) graduation that distilled them
    items: list[str] = Field(default_factory=list)  # items learned or reinforced
    text: str = ""
    graduations: int = 1  # more when later graduations folded into it
    folded: list[str] = Field(default_factory=list)  # the earlier graduations folded in


Record = Annotated[Exchange | Summary, Field(discriminator="kind")]
_RECORD: TypeAdapter[Exchange | Summary] = TypeAdapter(Record)


class SessionBuffer:
    def __init__(self, home: Path) -> None:
        self.root = home / SESSIONS_DIR
        self.version = 0  # bumped on every change; the web page polls it

    # ------------------------------------------------------------------ files

    def _dir(self, session: str) -> Path:
        return self.root / quote(session, safe="")

    def _path(self, session: str) -> Path:
        return self._dir(session) / CHAIN_FILE

    def sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(unquote(d.name) for d in self.root.iterdir() if (d / CHAIN_FILE).exists())

    def records(self, session: str) -> list[Exchange | Summary]:
        out: list[Exchange | Summary] = []
        try:
            fh = self._path(session).open(encoding="utf-8")
        except FileNotFoundError:
            return out
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    out.append(_RECORD.validate_python(json.loads(line)))
                except (json.JSONDecodeError, ValidationError):
                    log.warning("skipped a bad line in %s", self._path(session))
        return out

    def _write(self, session: str, records: Iterable[Exchange | Summary]) -> None:
        lines = "".join(r.model_dump_json() + "\n" for r in records)
        atomic_write(self._path(session), lines)
        self.version += 1

    # ------------------------------------------------------------------ reads

    def exchanges(self, session: str) -> list[Exchange]:
        """Exchanges not graduated yet: a graduation replaces the ones it covered with a
        summary."""
        return [r for r in self.records(session) if isinstance(r, Exchange)]

    def summaries(self, session: str) -> list[Summary]:
        return [r for r in self.records(session) if isinstance(r, Summary)]

    def last_activity(self, session: str) -> datetime | None:
        recs = self.records(session)
        return max((r.ts for r in recs), default=None)

    def iter_all(self) -> Iterator[tuple[str, list[Exchange | Summary]]]:
        for s in self.sessions():
            yield s, self.records(s)

    # ------------------------------------------------------------------ changes

    def append(self, session: str, ex: Exchange) -> Exchange:
        path = self._path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(ex.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.version += 1
        return ex

    def update(self, session: str, ex: Exchange) -> None:
        """Replace the exchange with the same id (a turn that went on after a background
        task finished)."""
        self._write(session, [ex if r.id == ex.id else r for r in self.records(session)])

    def drop_after(self, session: str, exchange_id: str) -> list[Exchange]:
        """The user rewound to this exchange: the exchanges after it were discarded and
        must not graduate. Returns them."""
        recs = self.records(session)
        idx = next((i for i, r in enumerate(recs) if r.id == exchange_id), None)
        if idx is None:
            return []
        dropped = [r for r in recs[idx + 1 :] if isinstance(r, Exchange)]
        if dropped:
            gone = {d.id for d in dropped}
            self._write(session, [r for r in recs if r.id not in gone])
        return dropped

    def remove(self, session: str, exchange_id: str) -> Exchange | None:
        recs = self.records(session)
        hit = next((r for r in recs if r.id == exchange_id and isinstance(r, Exchange)), None)
        if hit is not None:
            self._write(session, [r for r in recs if r.id != exchange_id])
        return hit

    def summarize(
        self, session: str, covered: Iterable[str], summary: Summary, into: str | None = None
    ) -> int:
        """Replace the covered exchanges with `summary`, placed where the first of them was;
        with `into`, the summary record of that id is replaced by `summary` instead (a fold)
        and the exchanges just go. Exchanges stored after them stay. Returns how many
        were replaced."""
        gone = set(covered)
        out: list[Exchange | Summary] = []
        placed = into is not None and any(r.id == into for r in self.records(session))
        replaced = 0
        for r in self.records(session):
            if placed and into is not None and r.id == into:
                out.append(summary)
                continue
            if isinstance(r, Exchange) and r.id in gone:
                replaced += 1
                if not placed:
                    out.append(summary)
                    placed = True
                continue
            out.append(r)
        if replaced:
            self._write(session, out)
        return replaced

    def replace_summaries(self, session: str, summary: Summary) -> int:
        """Put `summary` first in place of every summary record; exchanges stay. Returns
        how many summaries it replaced."""
        recs = self.records(session)
        keep = [r for r in recs if isinstance(r, Exchange)]
        self._write(session, [summary, *keep])
        return len(recs) - len(keep)

    def drop(self, session: str, covered: Iterable[str]) -> int:
        """Remove exchanges that could not be graduated (they were given up on)."""
        gone = set(covered)
        recs = self.records(session)
        keep = [r for r in recs if not (isinstance(r, Exchange) and r.id in gone)]
        if len(keep) != len(recs):
            self._write(session, keep)
        return len(recs) - len(keep)

    def delete(self, session: str) -> bool:
        d = self._dir(session)
        if not d.exists():
            return False
        shutil.rmtree(d)
        self.version += 1
        return True

    # ------------------------------------------------------------------ views

    def stats(self) -> dict[str, Any]:
        sessions = exchanges = summaries = size = 0
        for s in self.sessions():
            sessions += 1
            size += self._path(s).stat().st_size
            for r in self.records(s):
                if isinstance(r, Exchange):
                    exchanges += 1
                else:
                    summaries += 1
        return {"sessions": sessions, "exchanges": exchanges, "summaries": summaries, "bytes": size}
