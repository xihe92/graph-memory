"""Event log: one JSON line per thing the memory does, in ~/.graph-memory/events.jsonl.

It is the raw material for `gm stats` and the Memory help page. Every line has `ts` (UTC,
ISO 8601) and `kind`; the other fields depend on the kind:

    remember          session, exchange, project, continued, rewound
    rewind            session, dropped              (exchanges the user rewound away)
    forget            session, exchange, text       (a person removed one before it graduated)
    graduate          session, trigger, ok, segment, exchanges, items, new, calls, ms,
                      error, attempt, given_up, cleared
    buffer_dropped    session, exchanges, reason    (graduation gave up on them)
    session_cleared   session, trigger              (its folder went, nothing to learn)
    knowledge_recall  session, mode, prompt, items, scores, why, entities, seeds, reason, ms
    knowledge_use     session, mode, recalled, used, node
    knowledge_judge   session, items, labels, ok, ms, error   (a model's verdict per note)
    knowledge_dormant items
    knowledge_retract item, reason; knowledge_restore item
    request           method, ms, ok, error        (daemon side, hook methods only)
    hook              hook, ms, ok, error          (client side, the latency the agent feels)
    daemon            op ("start" | "stop"), pid;  ui  op, url | port, error

Writers append whole lines with O_APPEND, so the daemon and hook processes can share the
file. Only the daemon rotates it (events.jsonl -> events.jsonl.1) and prunes it: events
older than `events_days` are dropped once a day.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import clock
from .files import atomic_write

log = logging.getLogger(__name__)

EVENTS_FILE = "events.jsonl"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


def now_iso() -> str:
    return clock.now().isoformat(timespec="milliseconds")


class EventLog:
    """Append-only event sink. With `path=None` it only keeps the recent events in memory."""

    def __init__(
        self,
        path: Path | None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        rotate: bool = True,
        keep_recent: int = 500,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.rotate = rotate
        self.recent: deque[dict[str, Any]] = deque(maxlen=keep_recent)

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"ts": now_iso(), "kind": kind, **fields}
        self.recent.append(event)
        if self.path is not None:
            try:
                self._write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except OSError:
                log.exception("could not write event")
        return event

    def _write(self, line: str) -> None:
        assert self.path is not None
        if self.rotate:
            try:
                if self.path.stat().st_size >= self.max_bytes:
                    os.replace(self.path, self.path.with_name(self.path.name + ".1"))
            except FileNotFoundError:
                pass
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


def events_path(home: Path) -> Path:
    return home / EVENTS_FILE


def read_events(
    path: Path,
    *,
    since: datetime | None = None,
    kinds: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Events oldest first, from the rotated file and then the current one. Skips lines
    that don't parse (a crash mid-write leaves at most one)."""
    wanted = frozenset(kinds) if kinds is not None else None
    cutoff = since.isoformat() if since is not None else None
    for p in (path.with_name(path.name + ".1"), path):
        try:
            fh = open(p, encoding="utf-8")  # noqa: SIM115 - closed below
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                if wanted is not None and ev.get("kind") not in wanted:
                    continue
                if cutoff is not None and str(ev.get("ts", "")) < cutoff:
                    continue
                yield ev


def prune_events(path: Path, before: datetime) -> int:
    """Drop events older than `before`. The live file is moved aside first, so lines that
    hook processes append meanwhile go to a fresh file and are kept. Returns how many
    were dropped."""
    old = path.with_name(path.name + ".1")
    aside = path.with_name(path.name + ".pruning")
    if not _has_older(old if old.exists() else path, before.isoformat()):
        return 0  # nothing to drop: leave the files alone
    try:
        os.replace(path, aside)
    except FileNotFoundError:
        if not old.exists():
            return 0
    cutoff = before.isoformat()
    kept: list[str] = []
    dropped = 0
    for p in (old, aside):
        try:
            fh = open(p, encoding="utf-8")  # noqa: SIM115 - closed below
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    ts = str(json.loads(line).get("ts", ""))
                except (json.JSONDecodeError, AttributeError):
                    dropped += 1
                    continue
                if ts >= cutoff:
                    kept.append(line if line.endswith("\n") else line + "\n")
                else:
                    dropped += 1
    atomic_write(old, "".join(kept))
    with contextlib.suppress(FileNotFoundError):
        os.unlink(aside)
    return dropped


def _has_older(path: Path, cutoff: str) -> bool:
    """Whether the file's first event is older than `cutoff` (events are in time order)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        return str(json.loads(line).get("ts", "")) < cutoff
                    except (json.JSONDecodeError, AttributeError):
                        return True
    except FileNotFoundError:
        pass
    return False


def parse_since(text: str | None) -> datetime | None:
    """'30m', '24h', '7d', '2w', or 'all' (None)."""
    if not text or text == "all":
        return None
    units = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    num, unit = text[:-1], text[-1:]
    if unit not in units or not num.isdigit():
        raise ValueError(f"bad duration {text!r}: use e.g. 30m, 24h, 7d, 2w or all")
    return clock.now() - timedelta(**{units[unit]: int(num)})
