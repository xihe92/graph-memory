"""The graph memory daemon: one long-running local process per memory folder.

Protocol: one JSON request per connection on a Unix socket,
    {"method": "...", "params": {...}}\\n  ->  {"ok": true, "result": ...}\\n

Methods: ping, status, recall, remember, graduate, maintain, settled, forget, sessions,
knowledge, knowledge_item, knowledge_retract, knowledge_restore, shutdown.

What it keeps: the knowledge graph, for good; each live session's exchanges in the
session buffer, until they graduate; the event log. Recall reads only knowledge. The
event loop thread owns all three; model calls and embeddings run in worker threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import importlib
import json
import logging
import os
import random
import shutil
import signal
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from . import clock
from .adapters.base import Turn, mentioned_files
from .buffer import Exchange, SessionBuffer, Summary
from .config import Config
from .embedding import MODELS, LocalEmbedder
from .events import EventLog, events_path, prune_events, read_events
from .graduation import (
    SUMMARY_CHARS,
    Ask,
    GraduationError,
    apply,
    build_resummarize_prompt,
    extract_step,
    known_entities,
    known_topics,
    merge_step,
    parts,
    plan_merge,
    segment_input,
)
from .judge import build_judge_prompt, verdicts
from .knowledge import (
    TOPIC_PREFIX,
    Entity,
    Item,
    ItemEdge,
    KnowledgeStore,
    Segment,
    at_risk,
    in_view,
    named_topics,
    topic_of,
)
from .knowledge_recall import KnowledgeIndex, note, tokens, used_by
from .knowledge_recall import recall as knowledge_recall
from .models import CommandModel, claude_code_model
from .projects import project_of
from .redact import make_redactor, redacting
from .storage import json_note, sqlite_problem
from .vectors import Embedder, atomic_write

if TYPE_CHECKING:
    from .webui import UIServer

log = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 32 * 1024 * 1024
NOTHING_LEARNED = "Nothing worth keeping was learned from these exchanges."
NOTHING_PREFIX = "Nothing worth keeping"  # that placeholder, in any wording it had
TOPIC_RECORDS = 40  # the session's latest exchanges and summaries read for its topics
REBUILD = "rebuild"  # a session graduated again from its transcript
REBUILD_DIRECT_PARTS = 2  # a rebuild in more parts is previewed first unless forced
FOLDED = "folded"  # a session's summaries folded into one, its knowledge untouched
RESCOPE = "rescope"  # the model proposes where each item is recalled; a person applies it
RESCOPE_JOB = "memory:rescope"  # the queue's name for that job (it is no session)
RESCOPE_FILE = "rescope.json"
SUMMARIZE = "summarize"  # the model writes a brief of one scope's items for a person
SUMMARIZE_JOB = "memory:summarize:"  # + the scope: the queue's name for that job


def _rebuild_diff(
    base: dict[str, list[str]],
    store: KnowledgeStore,
    done: list[Segment],
    summary: str,
    session: str,
    exchanges: int,
) -> dict[str, Any]:
    """What a rebuild changed, against the item ids memory had before it."""
    before = set(base["items"])

    def brief(i: str) -> dict[str, Any]:
        it = store.items[i]
        return {"id": i, "type": it.type.value, "statement": it.statement, "valid": it.valid}

    touched = list(dict.fromkeys(i for g in done for i in g.items))
    return {
        "session": session,
        "exchanges": exchanges,
        "parts": len(done),
        "segments": [g.segment_id for g in done],
        "start": done[0].start.isoformat() if done else clock.now().isoformat(),
        "end": done[-1].end.isoformat() if done else clock.now().isoformat(),
        "summary": summary,
        "base": base,
        "new": [brief(i) for i in store.items if i not in before],
        "matched": [brief(i) for i in touched if i in before],
        "retired": [  # true before the rebuild, no longer true after it
            brief(i) for i in base["valid"] if i in store.items and not store.items[i].valid
        ],
    }


def _parse_time(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None
    except ValueError:
        return None


TIMED_METHODS = frozenset({"recall", "remember"})  # the ones a hook waits on
# A reply arriving this long after its prompt's recall is still checked for use after a
# daemon restart; older recalls are dropped (the turn was abandoned).
PENDING_USE_CHECK = timedelta(hours=6)
PROMPT_IN_EVENT = 200  # characters of the prompt kept on knowledge_recall events
# A session seen this recently, and not ended, is live: the Sessions page shows it even
# when nothing of it waits in the buffer (it all graduated, or its turn is in progress).
LIVE = timedelta(hours=24)
# ... but a session with nothing to show (nothing buffered or graduated, no prompt in
# progress) only this long: one that just started, not a script's run hours ago
JUST_STARTED = timedelta(minutes=30)
ENDING = frozenset({"session_end", "abandoned"})  # graduations after which the session goes


def _import(spec: str) -> Any:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def build_embedder(cfg: Config) -> Embedder | None:
    if cfg.embedder == "none":
        return None
    name = "all-MiniLM-L6-v2" if cfg.embedder in ("chroma-default", "default") else cfg.embedder
    if name in MODELS:
        return LocalEmbedder(name)
    factory: Callable[[], Embedder] = _import(cfg.embedder)
    return factory()


def build_model(cfg: Config) -> Ask | None:
    """The model graduation (and note judging) calls, or None without one."""
    if not cfg.graduation:
        return None
    model = cfg.graduation_model or cfg.model
    if cfg.model_backend == "claude-code":
        return claude_code_model(model, cfg.graduation_timeout, str(cfg.home)).complete
    if cfg.model_backend == "command" and cfg.model_command:
        return CommandModel(
            cfg.model_command, timeout=cfg.graduation_timeout, cwd=str(cfg.home)
        ).complete
    return None


@dataclass
class _Job:
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


class Daemon:
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
        self.grad_queue: asyncio.Queue[_Job] = asyncio.Queue()
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

    # ------------------------------------------------------------------ lifecycle

    async def serve(self) -> None:
        sock = self.cfg.socket_path
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()  # stale; we hold the lock, so no live daemon owns it
        server = await asyncio.start_unix_server(
            self._handle, path=str(sock), limit=MAX_REQUEST_BYTES
        )
        os.chmod(sock, 0o600)
        self._server = server
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self.stop)

        tasks = [
            asyncio.create_task(self._saver(), name="saver"),
            asyncio.create_task(self._idle_watch(), name="idle"),
            asyncio.create_task(self._graduator(), name="graduator"),
            asyncio.create_task(self._maintainer(), name="maintenance"),
        ]
        self.sweep(trigger="sweep")  # sessions left over from before this daemon started
        log.info("daemon listening on %s (pid %d)", sock, os.getpid())
        self.events.emit("daemon", op="start", pid=os.getpid(), storage=self.knowledge.backend.name)
        note = json_note(self.knowledge.backend.name)
        if note:
            log.warning(note)
        self._start_ui()
        self.ready.set()
        try:
            await self._stop.wait()
        finally:
            # No new requests (stop() already closed the listener), and the ones already
            # in finish before anything they could reach is closed: no caller sees a
            # half-stopped daemon or a dropped connection.
            server.close()
            await server.wait_closed()
            if self._handling:
                await asyncio.wait(set(self._handling), timeout=5)
            await self._stop_ui()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # everything once more, as insurance against a change that was never marked
            self.knowledge.save(full=True)
            self.knowledge.close()
            with contextlib.suppress(FileNotFoundError):
                sock.unlink()
            self.events.emit("daemon", op="stop", pid=os.getpid())
            log.info("daemon stopped")

    def stop(self) -> None:
        """Stop at once taking connections: a caller that comes now finds no daemon and
        starts a new one, which waits for this one to go (`acquire_lock`)."""
        if self._server is not None:
            self._server.close()
        self._stop.set()

    def _start_ui(self) -> None:
        """Serve the web page on `ui_port` while the daemon runs. A port already in use
        (another program, or a `gm ui --port` server) leaves the daemon without a page."""
        if not self.cfg.ui_port:
            return
        from .webui import UIServer

        try:
            self.ui = UIServer(self.cfg, "127.0.0.1", self.cfg.ui_port)
        except OSError as exc:
            log.warning("web page not served: port %d: %s", self.cfg.ui_port, exc)
            self.events.emit("ui", op="unavailable", port=self.cfg.ui_port, error=str(exc))
            return
        threading.Thread(target=self.ui.serve_forever, name="web-ui", daemon=True).start()
        log.info("web page on %s", self.ui.url)
        self.events.emit("ui", op="start", url=self.ui.url)

    async def _stop_ui(self) -> None:
        ui, self.ui = self.ui, None
        if ui is not None:
            await asyncio.to_thread(ui.shutdown)
            ui.server_close()

    # ------------------------------------------------------------------ requests

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # busy while answering, so the idle watch never stops the daemon mid-request; a
        # passive request (a web page polling) doesn't count as activity
        self._busy += 1
        active = True
        task = asyncio.current_task()
        if task is not None:
            self._handling.add(task)
        try:
            line = await reader.readline()
            started = time.perf_counter()
            method = "?"
            error: str | None = None
            try:
                req = json.loads(line)
                active = not req.get("passive")
                if active:
                    self._last_activity = time.monotonic()
                method = str(req["method"])
                result = self._dispatch(method, dict(req.get("params") or {}))
                resp: dict[str, Any] = {"ok": True, "result": result}
            except Exception as exc:  # report every failure to the caller
                log.exception("request failed")
                error = f"{type(exc).__name__}: {exc}"
                resp = {"ok": False, "error": error}
            if method in TIMED_METHODS or error is not None:
                self.events.emit(
                    "request",
                    method=method,
                    ms=round((time.perf_counter() - started) * 1000, 1),
                    ok=error is None,
                    error=error,
                )
            writer.write(json.dumps(resp).encode() + b"\n")
            await writer.drain()
        finally:
            if task is not None:
                self._handling.discard(task)
            self._busy -= 1
            if active:
                self._last_activity = time.monotonic()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _dispatch(self, method: str, p: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pid": os.getpid()}
        if method == "status":
            return self.status()
        if method == "recall":
            cwd = str(p["cwd"]) if p.get("cwd") else None
            prompt = self.redact(str(p["prompt"]))
            self._seen(str(p["session"]), project_of([], cwd) if cwd else None)
            self._asking[str(p["session"])] = (prompt, clock.now())
            context: dict[str, Any] = {}
            if self.cfg.knowledge_recall in ("shadow", "on"):
                lines = self.recall_knowledge(str(p["session"]), prompt, cwd)
                if lines and self.cfg.knowledge_recall == "on":
                    context["knowledge"] = lines
            return {"context": context}
        if method == "remember":
            return self.remember(str(p["session"]), dict(p["turn"]))
        if method == "graduate_now":
            return self.graduate_now(str(p["session"]))
        if method == "regraduate":
            return self.regraduate(str(p["session"]), bool(p.get("preview")), bool(p.get("force")))
        if method == "resummarize":
            return self.resummarize(str(p["session"]))
        if method == "rebuild_preview":
            return {k: v for k, v in self.rebuild_preview(str(p["session"])).items() if k != "base"}
        if method == "apply_rebuild":
            return self.apply_rebuild(str(p["session"]))
        if method == "graduate":
            if p.get("all"):
                return {"queued": self.sweep(min_idle=0.0, trigger="manual")}
            if p.get("trigger") == "compact":
                self.compacted(str(p["session"]))
            return {
                "queued": self.enqueue_graduation(
                    str(p["session"]),
                    str(p.get("trigger") or "manual"),
                    self.redact(str(p.get("compact_summary") or "")),
                )
            }
        if method == "maintain":
            return self.maintain()
        if method == "settled":
            return {"settled": self.settled()}
        if method == "forget":
            return self.forget(str(p["session"]), str(p["id"]))
        if method == "sessions":
            if p.get("since") == self.buffer.version and p.get("since") is not None:
                return {"version": self.buffer.version, "unchanged": True}
            return self.sessions_view()
        if method == "knowledge":
            return self.knowledge_view(lean=bool(p.get("lean")))
        if method == "knowledge_search":
            return self.knowledge_search(str(p.get("q") or ""))
        if method == "knowledge_item":
            return self.knowledge_detail(str(p["id"]))
        if method == "knowledge_retract":
            return self.knowledge_retract(str(p["id"]), str(p.get("reason") or ""))
        if method == "knowledge_restore":
            return self.knowledge_restore(str(p["id"]))
        if method == "knowledge_confirm":
            return self.knowledge_confirm(str(p["id"]))
        if method == "knowledge_delete":
            return self.knowledge_delete(
                [str(i) for i in p.get("ids") or []], str(p.get("reason") or "")
            )
        if method == "knowledge_undelete":
            return self.knowledge_undelete(str(p["batch"]))
        if method == "knowledge_review":
            return self.knowledge_review()
        if method == "rescope":
            return self.rescope()
        if method == "rescope_preview":
            return self.rescope_preview()
        if method == "apply_rescope":
            return self.apply_rescope()
        if method == "knowledge_scope":
            return self.knowledge_scope([str(i) for i in p.get("ids") or []], str(p["to"]))
        if method == "knowledge_summarize":
            return self.knowledge_summarize(str(p.get("scope") or ""), bool(p.get("force")))
        if method == "knowledge_summary":
            return self.knowledge_summary(str(p.get("scope") or ""))
        if method == "shutdown":
            self.stop()  # this answer still goes out: serve() waits for it
            return None
        raise ValueError(f"unknown method {method!r}")

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "ui": self.ui.url if self.ui is not None else None,
            "buffer": self.buffer.stats(),
            # the conversation graph from before the redesign is still on disk, or the
            # memory is still in the JSON files from before SQLite and can move
            "needs_migration": (self.cfg.home / "graph.json").exists()
            or (self.knowledge.backend.name == "json" and sqlite_problem() is None),
            "storage": self.knowledge.backend.name,
            "storage_note": json_note(self.knowledge.backend.name),
            "knowledge": {
                **self.knowledge.stats(),
                "graduation": self.ask is not None,
                "queued": self.grad_queue.qsize(),
                "graduating": self._graduating,
            },
        }

    # ------------------------------------------------------------------ remember

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

    def remember(self, session: str, turn_data: dict[str, Any]) -> dict[str, Any]:
        """Store the finished exchange in the session's buffer, then check which recalled
        notes the reply used."""
        turn = self._redacted(Turn.model_validate(turn_data))
        files = [*mentioned_files(turn.prompt), *turn.files]
        project = project_of(files, turn.cwd, use_cwd=False) or (
            project_of([], turn.cwd) if turn.cwd else None
        )
        self._seen(session, project)
        self._asking.pop(session, None)
        result: dict[str, Any] = {"rewound": 0, "continued": False}
        chain = self.buffer.exchanges(session)
        first = turn.source.first if turn.source else None
        same = next(
            (
                x
                for x in chain
                if first and x.source and x.source.first == first and x.prompt == turn.prompt
            ),
            None,
        )
        if same is not None:
            # the turn went on after a background task finished: the transcript reads
            # back as the same prompt with a longer reply
            ex = same.model_copy(
                update={
                    "reply": turn.reply,
                    "source": turn.source,
                    "files": turn.files or same.files,
                    "commands": turn.commands or same.commands,
                    "web": turn.web or same.web,
                }
            )
            if ex != same:
                self.buffer.update(session, ex)
            result["continued"] = True
        else:
            if turn.previous_reply_ref:
                # the prompt follows an older reply than the newest: the user rewound,
                # and what came after that reply was discarded
                prev = next(
                    (x for x in chain if x.source and x.source.last == turn.previous_reply_ref),
                    None,
                )
                if prev is not None:
                    dropped = self.buffer.drop_after(session, prev.id)
                    if dropped:
                        result["rewound"] = len(dropped)
                        self.events.emit("rewind", session=session, dropped=[d.id for d in dropped])
            ex = self.buffer.append(
                session,
                Exchange(
                    prompt=turn.prompt,
                    reply=turn.reply,
                    files=turn.files,
                    commands=turn.commands,
                    web=turn.web,
                    project=project,
                    source=turn.source,
                ),
            )
        result["exchange"] = ex.id
        result["project"] = project
        self.events.emit(
            "remember",
            session=session,
            exchange=ex.id,
            project=project,
            continued=result["continued"],
            rewound=result["rewound"],
        )
        recalled = self._recalled.pop(session, None)
        if recalled:
            self._background(self._check_use(session, recalled, turn.reply, ex.id))
            # a note that could have gone stale, or that nobody checked against a source,
            # is always judged; the rest, a sample
            stale_risk = any(
                at_risk(self.knowledge.items[i]) for i in recalled if i in self.knowledge.items
            )
            if (
                self.judge is not None
                and self.cfg.knowledge_recall == "on"
                and (stale_risk or self._rng() < self.cfg.judge_rate)
            ):
                self._background(self._judge(session, recalled, turn.prompt, turn.reply))
        return result

    def _background(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _judge(self, session: str, item_ids: list[str], prompt: str, reply: str) -> None:
        """A model labels each note shown for this turn useful, neutral, misleading or
        contradicted. A contradicted note is disputed: shown with a warning from then on,
        retracted when it happens again."""
        assert self.judge is not None
        items = [self.knowledge.items[i] for i in item_ids if i in self.knowledge.items]
        if not items:
            return
        started = time.perf_counter()
        try:
            answer = await asyncio.to_thread(self.judge, build_judge_prompt(prompt, reply, items))
            found = verdicts(answer, items)
            labels = {i: lab for i, (lab, _) in found.items()}
            self.events.emit(
                "knowledge_judge",
                session=session,
                items=[i.item_id for i in items],
                labels=labels,
                ms=round((time.perf_counter() - started) * 1000, 1),
                ok=True,
            )
            for item_id, (lab, why) in found.items():
                if lab != "contradicted" or item_id not in self.knowledge.items:
                    continue
                reason = why or f"contradicted by the reply to: {' '.join(prompt.split())[:80]}"
                it = self.knowledge.dispute(item_id, reason)
                self.knowledge_dirty = True
                self.events.emit(
                    "knowledge_dispute",
                    session=session,
                    item=item_id,
                    reason=reason,
                    disputes=it.disputes,
                    retracted=not it.valid,
                )
        except Exception as exc:
            log.warning("judging notes failed: %s", exc)
            self.events.emit(
                "knowledge_judge",
                session=session,
                items=[i.item_id for i in items],
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )

    def forget(self, session: str, exchange_id: str) -> dict[str, Any]:
        """Remove an exchange recorded by mistake, before it can graduate."""
        ex = self.buffer.remove(session, exchange_id)
        if ex is None:
            raise KeyError(f"no exchange {exchange_id!r} waiting in {session!r}")
        self.events.emit(
            "forget", session=session, exchange=exchange_id, text=" ".join(ex.prompt.split())[:120]
        )
        return {"exchange": exchange_id}

    # ------------------------------------------------------------------ background

    async def _saver(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.save_interval)
            if self.knowledge_dirty:
                self.knowledge_dirty = False
                try:
                    self.knowledge.save()
                except Exception:
                    self.knowledge_dirty = True
                    log.exception("knowledge save failed")

    async def _idle_watch(self) -> None:
        interval = min(15.0, max(0.05, self.cfg.idle_timeout / 4))
        while True:
            await asyncio.sleep(interval)
            idle = time.monotonic() - self._last_activity
            if (
                idle >= self.cfg.idle_timeout
                and self._busy == 0
                and self.grad_queue.empty()
                and self._graduating is None
                and not self._bg
            ):
                log.info("idle for %.0fs; shutting down", idle)
                self.stop()
                return

    async def _maintainer(self) -> None:
        interval = max(1.0, min(60.0, self.cfg.graduation_idle / 4))
        while True:
            await asyncio.sleep(interval)
            self.maintain()

    # ------------------------------------------------------------------ graduation

    def sweep(self, *, min_idle: float | None = None, trigger: str = "idle") -> int:
        """Queue the sessions whose newest exchange waited `min_idle` seconds (default:
        graduation_idle), and the sessions untouched for buffer_abandon_days."""
        now = clock.now()
        idle = self.cfg.graduation_idle if min_idle is None else min_idle
        queued = 0
        for session in self.buffer.sessions():
            recs = self.buffer.records(session)
            last = max((r.ts for r in recs), default=None)
            if last is None:
                self.buffer.delete(session)
                continue
            quiet = (now - last).total_seconds()
            if quiet >= self.cfg.buffer_abandon_days * 86400:
                queued += self.enqueue_graduation(session, "abandoned")
            elif any(isinstance(r, Exchange) for r in recs) and quiet >= idle:
                queued += self.enqueue_graduation(session, trigger)
        return queued

    def enqueue_graduation(self, session: str, trigger: str, compact_summary: str = "") -> bool:
        if self._grad_queued[session] and not compact_summary and trigger not in ENDING:
            return False  # one waiting already covers the same exchanges
        pending = self.buffer.exchanges(session)
        self._grad_queued[session] += 1
        self.grad_queue.put_nowait(
            _Job(session, trigger, compact_summary, pending[-1].id if pending else None)
        )
        return True

    def maintain(self) -> dict[str, int]:
        """One maintenance pass: graduate quiet sessions, clear abandoned ones, move
        knowledge that faded to dormant. Runs every minute or so; the simulator calls it
        directly after moving the clock."""
        queued = self.sweep()
        faded = self.knowledge.sweep()
        if faded:
            self.knowledge_dirty = True
            self.events.emit("knowledge_dormant", items=faded)
        return {"queued": queued, "dormant": len(faded), "events_pruned": self.prune_events()}

    def prune_events(self) -> int:
        """Keep `events_days` of the event log; runs at most once a day."""
        now = clock.now()
        path = self.events.path
        if (
            path is None
            or self.cfg.events_days <= 0  # 0: keep everything
            or (self._pruned is not None and now - self._pruned < timedelta(days=1))
        ):
            return 0
        self._pruned = now
        try:
            return prune_events(path, now - timedelta(days=self.cfg.events_days))
        except OSError:
            log.exception("pruning the event log failed")
            return 0

    def settled(self) -> bool:
        """No background work left: graduation, use checks."""
        return self.grad_queue.empty() and self._graduating is None and not self._bg

    async def _graduator(self) -> None:
        while True:
            job = await self.grad_queue.get()
            self._graduating = job.session
            try:
                await self._graduate(job)
            except Exception:
                log.exception("graduation bookkeeping failed for %s", job.session)
            finally:
                self._graduating = None
                self._grad_queued[job.session] -= 1
                if self._grad_queued[job.session] <= 0:
                    del self._grad_queued[job.session]
                self.grad_queue.task_done()

    async def _graduate(self, job: _Job) -> None:
        if job.trigger == REBUILD:
            await self._rebuild(job)
            return
        if job.trigger == FOLDED:
            await self._fold_summaries(job)
            return
        if job.trigger == RESCOPE:
            await self._rescope(job)
            return
        if job.trigger == SUMMARIZE:
            await self._brief(job)
            return
        session = job.session
        pending = self.buffer.exchanges(session)
        if job.upto is not None:
            idx = next((i for i, x in enumerate(pending) if x.id == job.upto), None)
            pending = pending[: idx + 1] if idx is not None else pending
        else:
            pending = []
        ending = job.trigger in ENDING
        if ending:
            self._live.pop(session, None)
            self._asking.pop(session, None)
        if not pending:
            if ending and not self.buffer.exchanges(session):
                self.buffer.delete(session)
                self.events.emit("session_cleared", session=session, trigger=job.trigger)
            return
        if self.ask is None:
            # no model: nothing can be learned, so nothing is kept past the session
            if ending:
                self.buffer.delete(session)
                self.events.emit("session_cleared", session=session, trigger=job.trigger)
            return
        # a compaction's own summary covers the whole session; otherwise the graduation
        # reads the summary the session's last graduation (or compaction) left
        earlier = ""
        if not job.compact_summary:
            last = self.buffer.summaries(session)
            if last and not last[-1].text.startswith(NOTHING_PREFIX):
                earlier = last[-1].text
        started = time.perf_counter()
        seg = segment_input(session, pending, job.trigger, job.compact_summary, earlier)
        try:
            known = known_entities(self.knowledge, seg)
            topics = known_topics(self.knowledge)
            ex = await asyncio.to_thread(extract_step, seg, self.ask, self.embedder, known, topics)
            plan = plan_merge(self.knowledge, seg, ex)
            merged = await asyncio.to_thread(merge_step, plan, self.ask)
            segment = apply(self.knowledge, plan, merged, started)
        except Exception as exc:
            self._graduation_failed(job, pending, exc, started)
            return
        # the knowledge is on disk before the exchanges leave the buffer: a crash in
        # between graduates them again, and merging removes the duplicates
        self.knowledge.save()
        self.knowledge_dirty = False
        self._failures.pop(session, None)
        covered = [x.id for x in pending]
        if ending and len(self.buffer.exchanges(session)) == len(covered):
            self.buffer.delete(session)
        else:
            self._summarize(job, pending, segment, ex.summary)
        self.events.emit(
            "graduate",
            session=session,
            trigger=job.trigger,
            segment=segment.segment_id,
            exchanges=segment.exchanges,
            items=len(segment.items),
            new=len(segment.new),
            calls=segment.calls,
            ms=segment.ms,
            ok=True,
            cleared=ending and not self.buffer.records(session),
        )
        log.info(
            "graduated %s: %d exchanges -> %d items", session, segment.exchanges, len(segment.items)
        )

    def _summarize(
        self, job: _Job, pending: list[Exchange], segment: Segment, written: str
    ) -> None:
        """Replace the graduated exchanges with a summary record: Claude's own after a
        compaction, else the summary of the session so far that the graduation wrote. A
        graduation folds into the summary just before it, so a session keeps one summary
        that grows, not one per pause; a compaction starts a new one."""
        session = job.session
        covered = [x.id for x in pending]
        recs = self.buffer.records(session)
        at = next((n for n, r in enumerate(recs) if r.id == covered[0]), 0)
        prev = recs[at - 1] if at > 0 else None
        if isinstance(prev, Summary) and prev.trigger != "compact" and job.trigger != "compact":
            folded = prev.model_copy(
                update={
                    "ts": clock.now(),
                    "trigger": job.trigger,
                    "end": pending[-1].ts,
                    "exchanges": prev.exchanges + len(pending),
                    "segment": segment.segment_id,
                    "items": list(dict.fromkeys([*prev.items, *segment.items])),
                    "text": written or prev.text,
                    "graduations": prev.graduations + 1,
                    "folded": [*prev.folded, *([prev.segment] if prev.segment else [])],
                }
            )
            self.buffer.summarize(session, covered, folded, into=prev.id)
            return
        text = job.compact_summary if job.trigger == "compact" else written
        if not text:  # a model that wrote no summary: what it learned, if anything
            learned = [self.knowledge.items[i] for i in segment.items if i in self.knowledge.items]
            text = "\n".join(f"[{i.type.value}] {i.statement}" for i in learned) or NOTHING_LEARNED
        self.buffer.summarize(
            session,
            covered,
            Summary(
                trigger=job.trigger,
                start=pending[0].ts,
                end=pending[-1].ts,
                exchanges=len(pending),
                segment=segment.segment_id,
                items=segment.items,
                text=text,
            ),
        )

    # ------------------------------------------------------------------ by hand

    def _session_ref(self, ref: str) -> str:
        """The buffered or live session an id, or an unambiguous prefix of it, names."""
        ref = ref.strip()
        known = set(self.buffer.sessions()) | set(self._live)
        hits = [s for s in known if ref and (s == ref or s.split(":", 1)[-1].startswith(ref))]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'ambiguous'} session {ref!r}")
        return hits[0]

    def graduate_now(self, ref: str) -> dict[str, Any]:
        """Graduate what a session has waiting now; the session goes on."""
        session = self._session_ref(ref)
        waiting = len(self.buffer.exchanges(session))
        queued = self.enqueue_graduation(session, "manual") if waiting else False
        return {"session": session, "waiting": waiting, "queued": queued}

    def regraduate(self, ref: str, preview: bool = False, force: bool = False) -> dict[str, Any]:
        """Queue a rebuild of what a session taught, read again from its transcript. With
        `preview` it runs on a copy of memory and changes nothing until `apply_rebuild`.
        A long session (more than REBUILD_DIRECT_PARTS parts) is always previewed first
        unless `force`: its replay is where stale items slip in."""
        from .adapters.claude_code import find_transcript

        sid = ref.strip().split(":", 1)[-1]
        path = find_transcript(sid)
        if path is None:
            raise KeyError(f"no transcript (or more than one) for session {ref!r}")
        session = f"claude-code:{path.stem}"
        self._grad_queued[session] += 1
        job = _Job(session, REBUILD, transcript=str(path), preview=preview, force=force)
        self.grad_queue.put_nowait(job)
        return {
            "session": session,
            "transcript": str(path),
            "queued": True,
            "preview": preview,
            "long_parts": REBUILD_DIRECT_PARTS,
        }

    def _preview_dir(self, session: str) -> Path:
        return self.cfg.home / "rebuild-preview" / quote(session, safe="")

    def _rebuild_input(self, job: _Job, turns: list[Any]) -> list[Exchange]:
        """The transcript's turns that already graduated and left the buffer: not the ones
        still waiting (they graduate as usual), not a turn still in progress."""
        session = job.session
        waiting = {x.source.first for x in self.buffer.exchanges(session) if x.source}
        asking = self._asking.get(session)
        xs = []
        for ts, t in turns:
            if t.source is not None and t.source.first in waiting:
                continue
            # after a restart the waiting prompt is known only as far as the log kept it
            said = " ".join(t.prompt.split())
            if asking is not None and said.startswith(" ".join(asking[0].split())):
                continue
            files = [*mentioned_files(t.prompt), *t.files]
            project = project_of(files, t.cwd, use_cwd=False) or (
                project_of([], t.cwd) if t.cwd else None
            )
            xs.append(
                Exchange(
                    prompt=t.prompt,
                    reply=t.reply,
                    ts=_parse_time(ts) or clock.now(),
                    files=t.files,
                    commands=t.commands,
                    web=t.web,
                    project=project,
                    source=t.source,
                )
            )
        return xs

    async def _rebuild(self, job: _Job) -> None:
        """Graduate a session again under the current rules, from the turns in its
        transcript, replayed in order: one graduation per part, each reading the summary
        the one before wrote, so a later decision retires an earlier one as it would have
        live. What the session taught before is matched (history too), not counted twice.

        For real, it changes memory part by part and then replaces the session's summaries
        with one. As a preview, it runs on a copy in `rebuild-preview/<session>/` and
        writes what it would change; `apply_rebuild` swaps the copy in."""
        from .adapters.claude_code import read_turns

        session, started = job.session, time.perf_counter()
        if self.ask is None or job.transcript is None:
            return
        turns = [
            (ts, self._redacted(t))
            for ts, t in await asyncio.to_thread(read_turns, Path(job.transcript))
        ]
        xs = self._rebuild_input(job, turns)
        todo = parts(xs)
        # a long session whose design moved on replays into many stale items: look first
        long = len(todo) > REBUILD_DIRECT_PARTS and not job.preview and not job.force
        preview = job.preview or long
        base = {
            "items": sorted(self.knowledge.items),
            "valid": sorted(i.item_id for i in self.knowledge.valid_items()),
            "segments": sorted(self.knowledge.segments),
        }
        if preview:
            pdir = self._preview_dir(session)
            shutil.rmtree(pdir, ignore_errors=True)
            pdir.mkdir(parents=True)
            self.knowledge.copy_to(pdir)  # saved first; then a copy to change freely
            self.knowledge_dirty = False
            store = KnowledgeStore(pdir, self.embedder)
        else:
            store = self.knowledge
        done: list[Segment] = []
        summary = ""
        try:
            for part in todo:
                seg = segment_input(session, part, REBUILD, earlier=summary)
                known = known_entities(store, seg)
                ex = await asyncio.to_thread(
                    extract_step, seg, self.ask, self.embedder, known, known_topics(store)
                )
                plan = plan_merge(store, seg, ex)
                merged = await asyncio.to_thread(merge_step, plan, self.ask)
                if plan.prompt is not None and merged is None:
                    # without the merge, what the session taught before would be added
                    # again as new items: stop here
                    raise GraduationError("the merge call failed twice")
                done.append(apply(store, plan, merged, time.perf_counter()))
                summary = ex.summary or summary
                store.save()
                if store is self.knowledge:
                    self.knowledge_dirty = False
        except Exception as exc:
            log.warning("rebuilding %s failed: %s", session, exc)
            self.events.emit(
                "graduate",
                session=session,
                trigger=REBUILD,
                ok=False,
                preview=preview,
                parts_done=len(done),
                error=f"{type(exc).__name__}: {exc}"[:300],
                ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return
        finally:
            if store is not self.knowledge:
                store.close()
        result = _rebuild_diff(base, store, done, summary, session, len(xs))
        if preview:
            (self._preview_dir(session) / "preview.json").write_text(json.dumps(result, indent=1))
        elif done:
            result["replaced"] = self._rebuild_summary(session, done, summary, xs)
        self.events.emit(
            "graduate",
            session=session,
            trigger=REBUILD,
            ok=True,
            preview=preview,
            long=long,
            exchanges=len(xs),
            parts=len(done),
            items=len({i for g in done for i in g.items}),
            new=len(result["new"]),
            retired=len(result["retired"]),
            calls=sum(g.calls for g in done),
            ms=round((time.perf_counter() - started) * 1000, 1),
            replaced=result.get("replaced", 0),
        )
        log.info("rebuilt %s%s: %d exchanges in %d parts", session,
                 " (preview)" if preview else "", len(xs), len(done))  # fmt: skip

    def _rebuild_summary(
        self, session: str, done: list[Segment], summary: str, xs: list[Exchange]
    ) -> int:
        """Replace the session's summary records with one for the whole rebuild; an ended
        session keeps no buffer."""
        if session not in self.buffer.sessions() or not done:
            return 0
        old = self.buffer.summaries(session)
        items = list(dict.fromkeys(i for g in done for i in g.items))
        learned = [self.knowledge.items[i] for i in items if i in self.knowledge.items]
        text = summary or "\n".join(f"[{i.type.value}] {i.statement}" for i in learned)
        return self.buffer.replace_summaries(
            session,
            Summary(
                trigger=REBUILD,
                start=xs[0].ts,
                end=xs[-1].ts,
                exchanges=len(xs),
                segment=done[-1].segment_id,
                items=items,
                text=text or NOTHING_LEARNED,
                graduations=len(done),
                # every earlier graduation of the session: the rebuild covers them all
                folded=list(
                    dict.fromkeys(
                        [g for s in old for g in (*s.folded, s.segment) if g]
                        + [
                            g.segment_id
                            for g in self.knowledge.segments.values()
                            if g.session == session and g.segment_id != done[-1].segment_id
                        ]
                    )
                ),
            ),
        )

    def resummarize(self, ref: str) -> dict[str, Any]:
        """Queue folding a session's summaries into one (see _fold_summaries)."""
        session = self._session_ref(ref)
        n = len(self.buffer.summaries(session))
        if n < 2:
            return {"session": session, "summaries": n, "queued": False}
        self._grad_queued[session] += 1
        self.grad_queue.put_nowait(_Job(session, FOLDED))
        return {"session": session, "summaries": n, "queued": True}

    async def _fold_summaries(self, job: _Job) -> None:
        """Replace a session's summary records with one, written by the model from what
        the session taught and the summaries along the way; knowledge is not touched. For
        a session whose summaries were written under older rules, when graduating it again
        would do more harm than good (a long session whose design changed as it went)."""
        session, started = job.session, time.perf_counter()
        old = self.buffer.summaries(session)
        if len(old) < 2:
            return
        k = self.knowledge
        taught = sorted(
            (i for i in k.valid_items() if any(s.session == session for s in i.sources)),
            key=lambda i: max(s.until for s in i.sources if s.session == session),
        )
        texts = [s.text for s in old if s.text and not s.text.startswith(NOTHING_PREFIX)]
        try:
            if self.ask is None:
                raise GraduationError("no model to write the summary")
            prompt = build_resummarize_prompt(taught, texts)
            text = " ".join((await asyncio.to_thread(self.ask, prompt)).split())[:SUMMARY_CHARS]
            if not text:
                raise GraduationError("the model wrote no summary")
        except Exception as exc:
            log.warning("folding the summaries of %s failed: %s", session, exc)
            self.events.emit("summaries_folded", session=session, ok=False, error=f"{exc}"[:300])
            return
        segs = [g for s in old for g in (*s.folded, s.segment) if g]
        segs += [g.segment_id for g in k.segments.values() if g.session == session]
        segs = list(dict.fromkeys(segs))
        self.buffer.replace_summaries(
            session,
            Summary(
                trigger=FOLDED,
                start=min((s.start for s in old if s.start), default=None),
                end=max((s.end for s in old if s.end), default=None),
                exchanges=sum(s.exchanges for s in old),
                segment=segs[-1] if segs else None,
                items=list(dict.fromkeys(i for s in old for i in s.items)),
                text=text,
                graduations=sum(s.graduations for s in old),
                folded=segs[:-1],
            ),
        )
        self.events.emit(
            "summaries_folded",
            session=session,
            ok=True,
            replaced=len(old),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def rebuild_preview(self, ref: str) -> dict[str, Any]:
        """What a previewed rebuild would change."""
        session = f"claude-code:{ref.strip().split(':', 1)[-1]}"
        hits = [
            p
            for p in (self.cfg.home / "rebuild-preview").glob("*/preview.json")
            if unquote(p.parent.name).split(":", 1)[-1].startswith(session.split(":", 1)[1])
        ]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'more than one'} rebuild preview for {ref!r}")
        return dict(json.loads(hits[0].read_text()))

    def apply_rebuild(self, ref: str) -> dict[str, Any]:
        """Swap a previewed rebuild in, if memory has not learned or lost anything since
        the preview started (counters such as uses since then are not carried over)."""
        result = self.rebuild_preview(ref)
        session = result["session"]
        k = self.knowledge
        if sorted(k.items) != result["base"]["items"] or (
            sorted(k.segments) != result["base"]["segments"]
        ):
            raise ValueError("memory changed since the preview: run the preview again")
        pdir = self._preview_dir(session)
        preview = KnowledgeStore(pdir, self.embedder)
        done = [preview.segments[g] for g in result["segments"] if g in preview.segments]
        # the preview's memory replaces this one: close it, copy the preview over it, open
        self.knowledge.close()
        preview.copy_to(self.cfg.home)
        preview.close()
        self.knowledge = KnowledgeStore(self.cfg.home, self.embedder)
        self.kindex = KnowledgeIndex(self.knowledge)
        self.knowledge_dirty = False
        xs = [
            Exchange(prompt="", reply="", ts=datetime.fromisoformat(t))
            for t in (result["start"], result["end"])
        ]
        result["replaced"] = self._rebuild_summary(session, done, result["summary"], xs)
        shutil.rmtree(pdir, ignore_errors=True)
        self.events.emit(
            "rebuild_applied", session=session, new=len(result["new"]), replaced=result["replaced"]
        )
        return {k2: v for k2, v in result.items() if k2 != "base"}

    def _graduation_failed(
        self, job: _Job, pending: list[Exchange], exc: Exception, started: float
    ) -> None:
        """Keep the exchanges for the next try, unless this session has failed too often
        or its oldest exchange is too old: then give up on them, so a broken model can't
        make the buffer grow without limit."""
        session = job.session
        self._failures[session] += 1
        age_days = (clock.now() - pending[0].ts).total_seconds() / 86400
        give_up = (
            self._failures[session] >= self.cfg.graduation_attempts
            or age_days >= self.cfg.buffer_max_days
        )
        log.warning("graduation failed for %s: %s", session, exc)
        self.events.emit(
            "graduate",
            session=session,
            trigger=job.trigger,
            ok=False,
            ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}"[:300],
            attempt=self._failures[session],
            given_up=give_up,
        )
        if give_up:
            self.buffer.drop(session, [x.id for x in pending])
            self._failures.pop(session, None)
            if job.trigger in ENDING and not self.buffer.exchanges(session):
                self.buffer.delete(session)
            self.events.emit(
                "buffer_dropped", session=session, exchanges=len(pending), reason=str(exc)[:200]
            )

    def compacted(self, session: str) -> None:
        """The session's context was compacted: what it was shown and what it learned
        before is no longer in its context, so both may be recalled again."""
        self._compacted[session] = clock.now()
        self._shown.pop(session, None)
        self._greeted.discard(session)
        self.events.emit("compacted", session=session)

    def _last_compaction(self, session: str) -> datetime | None:
        """In memory since this daemon started; else from the session's buffer, whose
        compaction summaries survive a restart."""
        if session in self._compacted:
            return self._compacted[session]
        times = [s.ts for s in self.buffer.summaries(session) if s.trigger == "compact"]
        return max(times) if times else None

    # ------------------------------------------------------------------ recall

    def recall_knowledge(self, session: str, prompt: str, cwd: str | None) -> list[str]:
        """Knowledge items for this prompt, as lines; logged either way. Leaves out items
        already shown in this session and items the session can still see (learned only
        from its own exchanges since its last compaction)."""
        try:
            k = self.knowledge
            shown = self._shown.setdefault(session, set())
            since = self._last_compaction(session)
            own = {i.item_id for i in k.valid_items() if in_view(i, session, since)}
            first = session not in self._greeted
            self._greeted.add(session)
            here = project_of([], cwd) if cwd else None
            topics = self._session_topics(session, prompt)
            got = knowledge_recall(
                k,
                self.kindex,
                prompt,
                project=here,
                exclude=shown | own,
                standing=first,
                topics=topics,
            )
            if self.cfg.knowledge_recall == "on":
                shown.update(i.item_id for i in got.items)
                k.mark_recalled(i.item_id for i in got.items)
            # replaced or cleared on every prompt: a turn interrupted before its reply
            # must not have its notes checked against the next prompt's reply
            if got.items:
                self._recalled[session] = [i.item_id for i in got.items]
            else:
                self._recalled.pop(session, None)
            self.events.emit(
                "knowledge_recall",
                session=session,
                mode=self.cfg.knowledge_recall,
                prompt=" ".join(prompt.split())[:PROMPT_IN_EVENT],
                items=[i.item_id for i in got.items],
                scores=got.scores,
                why=got.why,
                entities=got.entities,
                seeds=len(got.seeds),
                reason=got.reason,
                topics=sorted(t[len(TOPIC_PREFIX) :] for t in topics),
                ms=got.ms,
            )
            # each note says when it was last confirmed and where it came from; an item from
            # another project (an open item the prompt asked about) says which
            return [note(i, here) for i in got.items]
        except Exception:
            log.exception("knowledge recall failed")
            return []

    def _session_topics(self, session: str, prompt: str) -> set[str]:
        """The topics memory knows that this session is about: named in the prompt, in
        the exchanges it has waiting, or in its summaries."""
        self.kindex.fresh()
        scopes = self.kindex.topics()  # topics with something recall can show
        if not scopes:
            return set()
        texts = [prompt]
        for r in self.buffer.records(session)[-TOPIC_RECORDS:]:
            texts.append(r.text if isinstance(r, Summary) else f"{r.prompt}\n{r.reply[:3000]}")
        return named_topics(scopes, "\n".join(texts))

    async def _check_use(
        self, session: str, item_ids: list[str], reply: str, node: str | None = None
    ) -> None:
        """After the reply: which recalled items did it use? Those get stronger, and items
        used together get linked. Counted in shadow mode too: there it measures whether
        recall picked what the reply turned out to need."""
        try:
            embed = self.embedder
            vec = await asyncio.to_thread(embed, reply[:4000]) if embed else None
            k = self.knowledge
            words = set(tokens(reply))
            used = [
                i
                for i in item_ids
                if i in k.items and used_by(self.kindex, k.items[i], words, vec, k.vectors.get(i))
            ]
            k.mark_used(used)
            self.knowledge_dirty = True
            self.events.emit(
                "knowledge_use",
                session=session,
                mode=self.cfg.knowledge_recall,
                recalled=item_ids,
                used=used,
                node=node,
            )
        except Exception:
            log.exception("knowledge use check failed")

    # ------------------------------------------------------------------ views

    def sessions_view(self) -> dict[str, Any]:
        """For the Sessions page: each live session's chain, oldest first: what already
        graduated from it (each graduation, with the items it taught), then what waits in
        the buffer, then the prompt still waiting for its reply. And the sessions that
        graduated in the last three days."""
        k = self.knowledge
        now = clock.now()
        buffered = dict(self.buffer.iter_all())
        graduated_from = {g.session for g in k.segments.values()}
        keys = set(buffered) | {
            s
            for s, (seen, _) in self._live.items()
            if now - seen < JUST_STARTED
            or (now - seen < LIVE and (s in graduated_from or s in self._asking))
        }
        live = []
        for session in keys:
            recs = buffered.get(session, [])
            seen, known_project = self._live.get(session, (None, None))
            projects = Counter(r.project for r in recs if isinstance(r, Exchange) and r.project)
            graduated = sorted(
                (g for g in k.segments.values() if g.session == session),
                key=lambda g: g.start,
            )
            # a buffered summary already stands for its graduation
            summarized = {
                seg for r in recs if isinstance(r, Summary) for seg in (r.segment, *r.folded) if seg
            }
            chain: list[dict[str, Any]] = [
                {
                    "kind": "graduated",
                    "id": "g:" + g.segment_id,
                    "trigger": g.trigger,
                    "start": g.start.isoformat(),
                    "end": g.end.isoformat(),
                    "ts": g.created_at.isoformat(),
                    "exchanges": g.exchanges,
                    "items": [self._item_brief(i, g.segment_id) for i in g.items if i in k.items],
                }
                for g in graduated
                if g.segment_id not in summarized
            ]
            for r in recs:
                rec = r.model_dump(mode="json")
                if isinstance(r, Summary):  # what it learned, readable on the page
                    rec["items"] = [
                        self._item_brief(i, r.segment or "") for i in r.items if i in k.items
                    ]
                chain.append(rec)
            asking = self._asking.get(session)
            if asking is not None:
                chain.append(
                    {
                        "kind": "asking",
                        "id": "q:" + session,
                        "prompt": asking[0],
                        "ts": asking[1].isoformat(),
                    }
                )
            times = [x for x in (seen, *(r.ts for r in recs)) if x is not None]
            live.append(
                {
                    "session": session,
                    "project": projects.most_common(1)[0][0] if projects else known_project,
                    "last": max(times).isoformat() if times else None,
                    "waiting": sum(1 for r in recs if isinstance(r, Exchange)),
                    "records": chain,
                }
            )
        cutoff = clock.now() - timedelta(days=3)
        recent = sorted(
            (s for s in k.segments.values() if s.created_at >= cutoff),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return {
            "version": self.buffer.version,
            "sessions": sorted(live, key=lambda s: s["last"] or "", reverse=True),
            "graduated": [
                {
                    "segment": s.segment_id,
                    "session": s.session,
                    "trigger": s.trigger,
                    "exchanges": s.exchanges,
                    "at": s.created_at.isoformat(),
                    "items": [
                        {
                            "id": i,
                            "type": k.items[i].type.value,
                            "statement": k.items[i].statement,
                            "new": k.items[i].segments[0] == s.segment_id,
                        }
                        for i in s.items
                        if i in k.items
                    ],
                }
                for s in recent[:30]
            ],
        }

    def _item_brief(self, item_id: str, segment_id: str) -> dict[str, Any]:
        """An item as the Sessions page lists it under a graduation or summary."""
        it = self.knowledge.items[item_id]
        return {
            "id": item_id,
            "type": it.type.value,
            "statement": it.statement,
            "new": bool(it.segments) and it.segments[0] == segment_id,
            "valid": it.valid,
        }

    def _item_ref(self, ref: str) -> str:
        """The item id an id or unambiguous prefix names."""
        ref = ref.strip()
        hits = [i for i in self.knowledge.items if ref and i.startswith(ref)]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'ambiguous'} knowledge item {ref!r}")
        return hits[0]

    def knowledge_retract(self, ref: str, reason: str) -> dict[str, Any]:
        it = self.knowledge.retract(self._item_ref(ref), reason)
        self.knowledge_dirty = True
        self.events.emit("knowledge_retract", item=it.item_id, reason=it.invalid_reason)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def knowledge_restore(self, ref: str) -> dict[str, Any]:
        it = self.knowledge.restore(self._item_ref(ref))
        self.knowledge_dirty = True
        self.events.emit("knowledge_restore", item=it.item_id)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def rescope(self) -> dict[str, Any]:
        """Queue the model's pass over every valid item's scope (see rescope.py); it
        writes a preview and changes nothing until `apply_rescope`."""
        if self.ask is None:
            raise GraduationError("no model to ask")
        self._grad_queued[RESCOPE_JOB] += 1
        self.grad_queue.put_nowait(_Job(RESCOPE_JOB, RESCOPE))
        return {"queued": True, "items": sum(1 for _ in self.knowledge.valid_items())}

    async def _rescope(self, job: _Job) -> None:
        from .rescope import propose

        started = time.perf_counter()
        assert self.ask is not None
        try:
            changes = await asyncio.to_thread(propose, self.knowledge, self.ask)
        except Exception as exc:
            log.warning("rescoping failed: %s", exc)
            self.events.emit(RESCOPE, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            return
        plan = {"at": clock.now().isoformat(), "changes": changes}
        atomic_write(self.cfg.home / RESCOPE_FILE, json.dumps(plan, indent=1))
        self.events.emit(
            RESCOPE,
            ok=True,
            preview=True,
            changes=len(changes),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def rescope_preview(self) -> dict[str, Any]:
        path = self.cfg.home / RESCOPE_FILE
        if not path.exists():
            raise KeyError("no rescope preview: run `gm knowledge rescope` first")
        plan: dict[str, Any] = json.loads(path.read_text())
        plan["scopes"] = scope_names(
            [c["from"] for c in plan["changes"]] + [c["to"] for c in plan["changes"]]
        )
        return plan

    def apply_rescope(self) -> dict[str, Any]:
        """Apply the preview to the items still as it saw them (one changed by hand or by
        a graduation since keeps its scope)."""
        plan = self.rescope_preview()
        k = self.knowledge
        done, kept = [], []
        for c in plan["changes"]:
            it = k.items.get(c["id"])
            if it is None or it.scope != c["from"]:
                kept.append(c["id"])
                continue
            name = topic_of(c["to"])
            it.scope = (k.topic_key(name) or c["to"]) if name else c["to"]
            k.touch(c["id"])
            done.append(c["id"])
        k.save()
        self.knowledge_dirty = False
        (self.cfg.home / RESCOPE_FILE).unlink(missing_ok=True)
        self.events.emit(RESCOPE, ok=True, preview=False, changed=len(done), skipped=len(kept))
        return {"changed": len(done), "skipped": len(kept)}

    def knowledge_scope(self, refs: list[str], to: str) -> dict[str, Any]:
        """Set items' scope by hand: "global", "topic:<name>", or a project by its name
        or key."""
        k = self.knowledge
        ids = list(dict.fromkeys(self._item_ref(r) for r in refs))
        if not ids:
            raise KeyError("no items to scope")
        scope = self._scope_ref(to)
        for i in ids:
            k.items[i].scope = scope
        k.touch(*ids)
        k.save()
        self.knowledge_dirty = False
        self.events.emit("knowledge_scope", items=ids, scope=scope)
        return {"ids": ids, "scope": scope}

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

    # ------------------------------------------------------------------ summaries

    def _scope_items(self, scope: str) -> list[Item]:
        return [i for i in self.knowledge.valid_items() if i.scope == scope]

    def _scope_label(self, scope: str) -> tuple[str, str]:
        """(short name, how the prompt and the page call it)."""
        if scope == "global":
            return "global", "the user themself and how they like to work (global notes)"
        name = topic_of(scope)
        if name:
            return name, f"the topic {name}"
        names = {
            x["key"]: x["name"] for x in scope_names(i.scope for i in self.knowledge.items.values())
        }
        title = names.get(scope, scope.rsplit("/", 1)[-1])
        return title, f"the project {title}"

    def _summaries(self) -> dict[str, Any]:
        return self.knowledge.summaries()

    def knowledge_summarize(self, to: str, force: bool = False) -> dict[str, Any]:
        """Queue a brief of one scope's valid items (see summarize.py). Nothing is queued
        when the scope's brief is current, unless `force`, or when one is being written."""
        from .summarize import changed_since

        scope = self._scope_ref(to)
        items = self._scope_items(scope)
        if not items:
            raise KeyError(f"memory has no notes in {to!r} to summarize")
        if self.ask is None:
            raise GraduationError("no model to ask")
        job = SUMMARIZE_JOB + scope
        if self._grad_queued.get(job):
            return {"scope": scope, "queued": False, "running": True, "items": len(items)}
        have = self._summaries().get(scope)
        if have and not force and changed_since(have.get("notes") or {}, items) == 0:
            return {"scope": scope, "queued": False, "running": False, "items": len(items)}
        self._summary_errors.pop(scope, None)
        self._grad_queued[job] += 1
        self.grad_queue.put_nowait(_Job(job, SUMMARIZE, scope=scope))
        return {"scope": scope, "queued": True, "running": True, "items": len(items)}

    async def _brief(self, job: _Job) -> None:
        from .summarize import select, stamp, summarize

        scope = job.scope
        started = time.perf_counter()
        assert self.ask is not None
        items, left_out = select(self._scope_items(scope))
        name, label = self._scope_label(scope)
        try:
            if not items:
                raise KeyError("no notes left in this scope")
            brief = await asyncio.to_thread(summarize, label, items, self.ask)
        except Exception as exc:
            log.warning("summarizing %s failed: %s", scope, exc)
            self._summary_errors[scope] = f"{type(exc).__name__}: {exc}"[:300]
            self.events.emit(SUMMARIZE, ok=False, scope=scope, error=self._summary_errors[scope])
            return
        brief.update(
            scope=scope,
            name=name,
            at=clock.now().isoformat(),
            items=len(items),
            left_out=left_out,
            notes={i.item_id: stamp(i) for i in items},
        )
        valid = {i.scope for i in self.knowledge.valid_items()}
        kept = {k: v for k, v in self._summaries().items() if k in valid}  # drop empty scopes
        kept[scope] = brief
        self.knowledge.set_summaries(kept)
        self.knowledge_dirty = False
        self.events.emit(
            SUMMARIZE,
            ok=True,
            scope=scope,
            items=len(items),
            left_out=left_out,
            points=sum(len(s["points"]) for s in brief["sections"]),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def knowledge_summary(self, to: str) -> dict[str, Any]:
        """The cached brief of a scope, how many notes changed since it was written, and
        whether one is being written now."""
        from .summarize import changed_since

        scope = self._scope_ref(to)
        items = self._scope_items(scope)
        have = self._summaries().get(scope)
        return {
            "scope": scope,
            "name": self._scope_label(scope)[0],
            "items": len(items),
            "running": bool(self._grad_queued.get(SUMMARIZE_JOB + scope)),
            "error": self._summary_errors.get(scope),
            "summary": have,
            "changed": changed_since(have.get("notes") or {}, items) if have else 0,
        }

    def knowledge_confirm(self, ref: str) -> dict[str, Any]:
        """A person checked the item and it still holds."""
        it = self.knowledge.confirm(self._item_ref(ref))
        self.knowledge_dirty = True
        self.events.emit("knowledge_confirm", item=it.item_id)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def knowledge_delete(self, refs: list[str], reason: str) -> dict[str, Any]:
        """Delete items for good (see KnowledgeStore.delete); undone by its batch id."""
        ids = list(dict.fromkeys(self._item_ref(r) for r in refs))
        if not ids:
            raise KeyError("no items to delete")
        batch = self.knowledge.delete(ids, reason)
        self.knowledge.save()  # the log of the deletion and the store agree on disk
        self.knowledge_dirty = False
        self.events.emit(
            "knowledge_delete",
            batch=batch["batch"],
            items=ids,
            bridges=len(batch["bridges"]),
            entities=len(batch["entities"]),
            reason=reason,
        )
        return {
            "batch": batch["batch"],
            "deleted": [r["item"]["statement"] for r in batch["items"]],
            "bridges": len(batch["bridges"]),
            "entities": [r["entity"]["name"] for r in batch["entities"]],
        }

    def knowledge_undelete(self, batch_id: str) -> dict[str, Any]:
        batch = self.knowledge.undelete(batch_id.strip())
        self.knowledge.save()
        self.knowledge_dirty = False
        self.events.emit("knowledge_undelete", batch=batch["batch"])
        return {"batch": batch["batch"], "restored": len(batch["items"])}

    def knowledge_review(self) -> dict[str, Any]:
        """What a person should look at: disputed items, items past their review date,
        and the deletions that can still be undone."""
        now = clock.now()
        items = []
        for i in self.knowledge.valid_items():
            if i.disputes:
                why = f"disputed: {i.dispute_reason}" if i.dispute_reason else "disputed"
            elif i.overdue(now):
                why = f"not re-checked since {i.checked.date().isoformat()}"
            else:
                continue
            items.append(
                {
                    "id": i.item_id,
                    "type": i.type.value,
                    "statement": i.statement,
                    "why": why,
                    "recalled": i.recallable,
                }
            )
        return {
            "items": items,
            "deleted": [
                {
                    "batch": b["batch"],
                    "at": b["at"],
                    "reason": b["reason"],
                    "statements": [r["item"]["statement"] for r in b["items"]],
                }
                for b in reversed(self.knowledge.deleted())
            ],
        }

    def knowledge_view(self, lean: bool = False) -> dict[str, Any]:
        """Every item, entity and edge. `lean` (the Knowledge page) leaves out what only an
        item's detail shows (reason, sources, provenance, review dates): the page loads
        those on click, by id, and searches through the daemon."""
        k = self.knowledge
        now = clock.now()

        def item(i: Item) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": i.item_id,
                "type": i.type.value,
                "statement": i.statement,
                "scope": i.scope,
                "entities": i.entities,
                "valid": i.valid,
                "strength": i.strength,
                "area": i.area.value,
                "uses": i.uses,
                "created_at": i.created_at.isoformat(),
                "invalid_at": i.invalid_at.isoformat() if i.invalid_at else None,
                "invalidated_by": i.invalidated_by,
                "invalid_reason": i.invalid_reason,
                "disputes": i.disputes,
                "overdue": i.overdue(now),
            }
            if not lean:
                out.update(
                    why=i.why,
                    vitality=round(i.vitality(now), 2),
                    sources=[s.model_dump(mode="json") for s in i.sources],
                    origin=i.origin.value,
                    ref=i.ref,
                    applies_to=i.applies_to,
                    volatility=i.volatility.value,
                    checked=i.checked.isoformat(),
                    dispute_reason=i.dispute_reason,
                    recallable=i.recallable,
                )
            return out

        def entity(e: Entity) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": e.entity_id,
                "name": e.name,
                "kind": e.kind.value,
                "mentions": e.mentions,
            }
            if not lean:
                out["summary"] = e.summary
            return out

        def edge(e: ItemEdge) -> dict[str, Any]:
            out: dict[str, Any] = {
                "source": e.source,
                "target": e.target,
                "relation": e.relation.value,
                "weight": e.weight,
                "reason": e.reason,
            }
            if not lean:
                out["origin"] = e.origin
            return out

        return {
            "version": k.version,
            "stats": k.stats(),
            "scopes": scope_names(i.scope for i in k.items.values()),
            "entities": [entity(e) for e in k.entities.values()],
            "items": [item(i) for i in k.items.values()],
            "edges": [edge(e) for e in k.edges.values()],
        }

    def knowledge_search(self, q: str) -> dict[str, Any]:
        """Items whose statement or reason contains the words, case aside, or that are about
        an entity whose name does; and those entities. What the Knowledge page's search
        box highlights."""
        q = q.strip().lower()
        k = self.knowledge
        if not q:
            return {"q": q, "items": [], "entities": []}
        ents = {e.entity_id for e in k.entities.values() if q in e.name.lower()}
        items = [
            i.item_id
            for i in k.items.values()
            if q in f"{i.statement} {i.why}".lower() or ents.intersection(i.entities)
        ]
        return {"q": q, "items": items, "entities": sorted(ents)}

    def knowledge_detail(self, ref: str) -> dict[str, Any]:
        k = self.knowledge
        iid = next((i for i in k.items if ref and i.startswith(ref)), None)
        if iid is None:
            eid = next((e for e in k.entities if ref and e.startswith(ref)), None)
            if eid is None:
                raise KeyError(ref)
            e = k.entities[eid]
            return {
                "version": k.version,
                "entity": e.model_dump(mode="json"),
                "items": [i.model_dump(mode="json") for i in k.items.values() if eid in i.entities],
            }
        item = k.items[iid]
        due = item.review_due()
        return {
            "version": k.version,
            "item": item.model_dump(mode="json"),
            "vitality": round(item.vitality(), 3),
            "checked": item.checked.isoformat(),
            "review_due": due.isoformat() if due else None,
            "overdue": item.overdue(),
            "recallable": item.recallable,
            "entities": [
                k.entities[e].model_dump(mode="json") for e in item.entities if e in k.entities
            ],
            "neighbours": [
                {
                    "id": other,
                    "relation": e.relation.value,
                    "weight": e.weight,
                    "origin": e.origin,
                    "reason": e.reason,
                    "outgoing": e.source == iid,
                    "statement": k.items[other].statement,
                    "valid": k.items[other].valid,
                }
                for other, e in k.neighbours(iid)
                if other in k.items
            ],
            # how it was learned: each graduation that stated it (the exchanges are gone)
            "learned": [
                {
                    "session": s.session,
                    "trigger": s.trigger,
                    "exchanges": s.exchanges,
                    "from": s.start.isoformat(),
                    "to": s.end.isoformat(),
                    "at": s.created_at.isoformat(),
                }
                for s in (k.segments[sid] for sid in item.segments if sid in k.segments)
            ],
        }


def scope_names(scopes: Iterable[str]) -> list[dict[str, Any]]:
    """Each scope with a short display name and its item count: global, then projects,
    then topics. A project is named by its directory, with parent directories added
    until the names are unique; a topic by its name."""
    counts = Counter(scopes)
    topics = sorted((s for s in counts if s.startswith(TOPIC_PREFIX)), key=str.lower)
    projects = sorted(s for s in counts if s != "global" and s not in topics)
    parts = {s: [x for x in s.removeprefix("repo:").split("/") if x] or [s] for s in projects}
    depth = dict.fromkeys(projects, 1)
    while True:
        names = {s: "/".join(parts[s][-depth[s] :]) for s in projects}
        clash = [s for s in projects if list(names.values()).count(names[s]) > 1]
        grow = [s for s in clash if depth[s] < len(parts[s])]
        if not grow:
            break
        for s in grow:
            depth[s] += 1
    out = (
        [{"key": "global", "name": "global", "items": counts["global"]}]
        if "global" in counts
        else []
    )
    return (
        out
        + [{"key": s, "name": names[s], "items": counts[s]} for s in projects]
        + [
            {"key": s, "name": s[len(TOPIC_PREFIX) :], "items": counts[s], "topic": True}
            for s in topics
        ]
    )


def _answers(sock: Path) -> bool:
    """Whether a daemon is accepting connections on `sock`."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(str(sock))
        except OSError:
            return False
    return True


def acquire_lock(cfg: Config, wait: float) -> IO[str] | None:
    """Take the daemon lock, held for the process lifetime. A daemon that is still
    shutting down holds it for a moment after it stops answering: wait up to `wait`
    seconds for it to go, so a hook that came in during that moment gets a daemon
    instead of a timeout. A daemon that answers is running: give up at once."""
    lock = open(cfg.lock_path, "a")  # noqa: SIM115 - "a": don't wipe the holder's pid
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if _answers(cfg.socket_path) or time.monotonic() >= deadline:
                lock.close()
                return None
            time.sleep(0.05)
    lock.truncate(0)
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


def run_daemon(cfg: Config) -> int:
    """Entry point for `gm daemon`. Exits quietly if another daemon is running."""
    logging.basicConfig(
        filename=str(cfg.log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    lock = acquire_lock(cfg, wait=cfg.start_timeout)
    if lock is None:
        log.info("another daemon is running; exiting")
        return 0

    embedder = build_embedder(cfg)
    if embedder is not None:
        embedder("warm up")  # load the model before accepting requests
    daemon = Daemon(
        cfg,
        knowledge=KnowledgeStore(cfg.home, embedder),
        embedder=embedder,
        graduation_model=build_model(cfg),
    )
    asyncio.run(daemon.serve())
    return 0
