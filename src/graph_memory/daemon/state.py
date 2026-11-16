"""What every part of the daemon shares: its state, set up once and seeded from the
event log, and the helpers more than one part uses. The event loop thread owns all of
it, so none of it is locked.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from .. import clock
from ..adapters.base import Turn
from ..buffer import SessionBuffer
from ..config import Config
from ..events import EventLog, events_path, read_events
from ..graduation import Ask
from ..knowledge import TOPIC_PREFIX, KnowledgeStore
from ..knowledge_recall import KnowledgeIndex
from ..redact import make_redactor, redacting
from ..vectors import Embedder
from .scopes import scope_names

if TYPE_CHECKING:
    from ..webui import UIServer

log = logging.getLogger("graph_memory.daemon")  # one log for the whole daemon

NOTHING_LEARNED = "Nothing worth keeping was learned from these exchanges."
NOTHING_PREFIX = "Nothing worth keeping"  # that placeholder, in any wording it had

# what a queued job does, besides graduating a session's waiting exchanges (Job.trigger)
REBUILD = "rebuild"  # a session graduated again from its transcript
FOLDED = "folded"  # a session's summaries folded into one, its knowledge untouched
RESCOPE = "rescope"  # the model proposes where each item is recalled; a person applies it
SUMMARIZE = "summarize"  # the model writes a brief of one scope's items for a person
ENDING = frozenset({"session_end", "abandoned"})  # graduations after which the session goes

# A reply arriving this long after its prompt's recall is still checked for use after a
# daemon restart; older recalls are dropped (the turn was abandoned).
PENDING_USE_CHECK = timedelta(hours=6)
# A session seen this recently, and not ended, is live: the Sessions page shows it even
# when nothing of it waits in the buffer (it all graduated, or its turn is in progress).
LIVE = timedelta(hours=24)


@dataclass
class Job:
    """A graduation waiting to run. `upto`: the newest exchange it covers, fixed when it
    was queued, so exchanges stored while it runs wait for the next one."""

    session: str
    trigger: str
    compact_summary: str = ""
    upto: str | None = None
    transcript: str | None = None  # a rebuild reads the session's turns from here
    preview: bool = False  # a rebuild on a copy of memory, to look at before applying
    force: bool = False  # a long rebuild changes memory directly, without a preview first
    scope: str = ""  # a summary's scope


class DaemonState:
    """The daemon's state, and the helpers more than one part of it uses. Each
    part (turns.py, graduating.py, ...) is a class on top of this one; Daemon
    (core.py) puts them together."""

    def __init__(
        self,
        cfg: Config,
        *,
        knowledge: KnowledgeStore | None = None,
        embedder: Embedder | None = None,
        graduation_model: Ask | None = None,
        events: EventLog | None = None,
        buffer: SessionBuffer | None = None,
        judge_model: Ask | None = None,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.events = events if events is not None else EventLog(events_path(cfg.home))
        self.knowledge = knowledge or KnowledgeStore(cfg.home, embedder)
        self.buffer = buffer or SessionBuffer(cfg.home)
        # secrets are replaced as text arrives, and in what models answer: see redact.py
        self.redact = make_redactor(cfg.redact, cfg.redact_patterns)
        self.ask = redacting(graduation_model, self.redact)
        self.judge = redacting(
            judge_model if judge_model is not None else graduation_model, self.redact
        )
        self._rng = rng
        # graduations run one at a time, so the knowledge store has a single writer and
        # merges see each other's results
        self.grad_queue: asyncio.Queue[Job] = asyncio.Queue()
        self._grad_queued: Counter[str] = Counter()  # session -> graduations waiting
        self._summary_errors: dict[str, str] = {}  # scope -> why its last brief failed
        self._graduating: str | None = None
        self._failures: Counter[str] = Counter()  # session -> failed graduations in a row
        self.knowledge_dirty = False
        self.kindex = KnowledgeIndex(self.knowledge)
        self._shown: dict[str, set[str]] = {}  # session -> knowledge items already shown
        # session -> items recalled for the prompt, until its reply is checked for use
        self._recalled: dict[str, list[str]] = self._pending_use_checks()
        # session -> (last seen, project); the prompt waiting for its reply, per session
        self._live: dict[str, tuple[datetime, str | None]] = {}
        self._asking: dict[str, tuple[str, datetime]] = {}
        self._seed_live()
        # sessions that already had their first prompt since starting or compacting
        self._greeted: set[str] = set()
        self._compacted: dict[str, datetime] = {}  # session -> time of its last compaction
        self._seed_shown()
        self._bg: set[asyncio.Task[None]] = set()
        self._stop = asyncio.Event()
        self._server: asyncio.AbstractServer | None = None
        self._handling: set[asyncio.Task[Any]] = set()  # requests being answered
        self._last_activity = time.monotonic()
        self._busy = 0
        self._pruned: datetime | None = None  # when the event log was last pruned
        self.ready = asyncio.Event()
        self.ui: UIServer | None = None  # the web page, served while the daemon runs

    def _pending_use_checks(self, within: timedelta = PENDING_USE_CHECK) -> dict[str, list[str]]:
        """Recalls still waiting for their reply when the last daemon stopped, from the
        event log, so a restart between a prompt and its reply doesn't lose the check."""
        path = self.events.path
        if path is None:
            return {}
        pending: dict[str, list[str]] = {}
        kinds = ("knowledge_recall", "knowledge_use")
        for ev in read_events(path, since=clock.now() - within, kinds=kinds):
            session = str(ev.get("session"))
            items = [str(i) for i in ev.get("items") or []]
            if ev["kind"] == "knowledge_recall" and items:
                pending[session] = items
            else:
                pending.pop(session, None)
        return pending

    def _seed_live(self, within: timedelta = LIVE) -> None:
        """Sessions active in the last day and not ended, and a prompt still waiting for
        its reply, from the event log, so a restart doesn't hide them from the Sessions
        page (and a rebuild doesn't read a turn that hasn't finished)."""
        path = self.events.path
        if path is None:
            return
        kinds = ("knowledge_recall", "remember", "graduate", "session_cleared")
        for ev in read_events(path, since=clock.now() - within, kinds=kinds):
            session = str(ev.get("session"))
            kind = ev["kind"]
            if kind in ("knowledge_recall", "remember"):
                prev = self._live.get(session)
                project = ev.get("project") or (prev[1] if prev else None)
                at = datetime.fromisoformat(str(ev["ts"]))
                self._live[session] = (at, project)
                if kind == "knowledge_recall" and ev.get("prompt"):
                    self._asking[session] = (str(ev["prompt"]), at)  # clipped in the log
                else:
                    self._asking.pop(session, None)
            elif kind == "session_cleared" or (
                kind == "graduate" and ev.get("ok") and ev.get("trigger") in ENDING
            ):
                self._live.pop(session, None)
                self._asking.pop(session, None)

    def _seed_shown(self) -> None:
        """What each session was already shown, and whether it had its first prompt,
        from the event log: the daemon stops when idle, and a restart must not show a
        session the same notes and standing preferences again. A compaction starts
        both over; an ended session is forgotten."""
        path = self.events.path
        if path is None:
            return
        kinds = ("knowledge_recall", "compacted", "graduate", "session_cleared")
        since = clock.now() - timedelta(days=self.cfg.buffer_abandon_days)
        for ev in read_events(path, since=since, kinds=kinds):
            session = str(ev.get("session"))
            kind = ev["kind"]
            if kind == "knowledge_recall":
                self._greeted.add(session)
                if ev.get("mode") == "on":
                    self._shown.setdefault(session, set()).update(
                        str(i) for i in ev.get("items") or []
                    )
            elif kind == "compacted":
                self._shown.pop(session, None)
                self._greeted.discard(session)
                self._compacted[session] = datetime.fromisoformat(str(ev["ts"]))
            elif kind == "session_cleared" or (
                kind == "graduate" and ev.get("ok") and ev.get("trigger") in ENDING
            ):
                self._shown.pop(session, None)
                self._greeted.discard(session)
                self._compacted.pop(session, None)

    def _seen(self, session: str, project: str | None = None) -> None:
        prev = self._live.get(session)
        self._live[session] = (clock.now(), project or (prev[1] if prev else None))

    # ------------------------------------------------------------------ helpers

    def _session_ref(self, ref: str) -> str:
        """The buffered or live session an id, or an unambiguous prefix of it, names."""
        ref = ref.strip()
        known = set(self.buffer.sessions()) | set(self._live)
        hits = [s for s in known if ref and (s == ref or s.split(":", 1)[-1].startswith(ref))]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'ambiguous'} session {ref!r}")
        return hits[0]

    def _scope_ref(self, to: str) -> str:
        """The scope key a person means: "global", "topic:<name>" (in the spelling memory
        has), or a project by its name or key."""
        k = self.knowledge
        want = to.strip()
        scope: str | None
        if want.lower() == "global":
            scope = "global"
        elif want.lower().startswith(TOPIC_PREFIX):
            scope = k.topic_key(want[len(TOPIC_PREFIX) :])
        else:
            repos = [
                x for x in scope_names(i.scope for i in k.items.values()) if not x.get("topic")
            ]
            hits = [
                x["key"] for x in repos if want in (x["key"], x["name"]) and x["key"] != "global"
            ]
            scope = hits[0] if len(hits) == 1 else None
        if not scope:
            raise ValueError(
                f"unknown scope {to!r}: use global, topic:<name>, or a project memory knows"
            )
        return scope

    def _redacted(self, turn: Turn) -> Turn:
        """The turn with secrets replaced in everything the daemon keeps or sends on."""
        r = self.redact
        return turn.model_copy(
            update={
                "prompt": r(turn.prompt),
                "reply": r(turn.reply),
                "commands": r.many(turn.commands),
                "web": r.many(turn.web),
                "subagents": [
                    s.model_copy(update={"task": r(s.task), "result": r(s.result)})
                    for s in turn.subagents
                ],
            }
        )
