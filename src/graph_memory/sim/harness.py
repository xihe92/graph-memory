"""The simulator: replays a scenario against a real daemon through the real hooks.

    Simulator(scenario).run() -> Report

For each turn it does what Claude Code does: fires the UserPromptSubmit hook (`gm recall`)
and keeps what the hook prints (the context the agent would see), appends the prompt,
tool calls and reply to a JSONL transcript in Claude Code's format, then fires the Stop
hook (`gm remember`), which reads the turn back from that transcript. Compaction writes
a compact summary into the transcript and fires SessionStart(compact) (`gm graduate`);
ending a session fires SessionEnd. Time is simulated (graph_memory.clock): `advance`
moves it and runs the daemon's maintenance, so idle graduation and decay happen as they
would over days.

The daemon runs in a thread of this process with an isolated home, socket, embedder and
graduation model: the scenario's oracle (default) or a real model.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .. import cli, clock
from ..client import DaemonUnavailable, call
from ..config import Config, load_config
from ..daemon import Daemon
from ..events import EventLog, events_path
from ..knowledge import Item, KnowledgeStore
from ..vectors import Embedder
from .embed import hash_embed
from .oracle import OracleModel, kind_of
from .scenario import (
    Advance,
    Assert,
    Check,
    Compact,
    End,
    Expect,
    FailNext,
    Probe,
    Restart,
    Retract,
    Scenario,
    Step,
    Turn,
)

KNOWLEDGE_HEADER = "Knowledge from earlier sessions that may apply here:"


def note_statement(line: str) -> str:
    """The statement of a recalled note: without its [type, date, origin] tag, its reason,
    and the project, dispute or review notes after it."""
    body = line.split("] ", 1)[1] if line.startswith("[") and "] " in line else line
    for cut in (" (why: ", " [project: ", " [disputed ", " [not re-checked"):
        at = body.find(cut)
        body = body[:at] if at >= 0 else body
    return body


AGENT = "claude-code"


@dataclass
class Outcome:
    step: int
    kind: str
    label: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tally:
    """Recall and use-detection counts over the checked steps."""

    checks: int = 0
    hits: int = 0
    misses: int = 0
    forbidden: int = 0
    extras: int = 0
    shown: int = 0
    empty_ok: int = 0
    empty_violations: int = 0
    use_tp: int = 0
    use_fn: int = 0
    use_fp: int = 0

    def summary(self) -> dict[str, Any]:
        shown_scored = self.hits + self.forbidden + self.extras
        return {
            "checks": self.checks,
            "required_shown": f"{self.hits}/{self.hits + self.misses}",
            "recall": round(self.hits / (self.hits + self.misses), 3)
            if self.hits + self.misses
            else None,
            "precision": round(self.hits / shown_scored, 3) if shown_scored else None,
            "forbidden_shown": self.forbidden,
            "noise_shown": self.extras,
            "silence_ok": f"{self.empty_ok}/{self.empty_ok + self.empty_violations}",
            "use_detection": {"tp": self.use_tp, "fn": self.use_fn, "fp": self.use_fp},
        }


@dataclass
class Report:
    scenario: str
    description: str
    model: str
    embedder: str
    recall_mode: str
    outcomes: list[Outcome]
    metrics: dict[str, Any]
    home: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)  # the final store, for review

    @property
    def ok(self) -> bool:
        return all(o.ok for o in self.outcomes)

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.ok]

    def text(self, verbose: bool = False) -> str:
        checked = [o for o in self.outcomes if o.kind in ("turn", "probe", "check", "assert")]
        passed = sum(o.ok for o in self.outcomes)
        lines = [
            f"== {self.scenario} [{self.model}/{self.embedder}/{self.recall_mode}] "
            f"{'PASS' if self.ok else 'FAIL'}  {passed}/{len(self.outcomes)} steps ok, "
            f"{len(checked)} with checks",
        ]
        if self.description:
            lines.append(f"   {self.description}")
        for o in self.outcomes:
            if verbose or not o.ok:
                mark = "ok  " if o.ok else "FAIL"
                label = f" {o.label}" if o.label else ""
                lines.append(f"   {mark} #{o.step:<3} {o.kind}{label}: {o.detail}")
        m = self.metrics
        r = m["recall"]
        lines.append(
            f"   recall: required {r['required_shown']}, precision {r['precision']}, "
            f"forbidden {r['forbidden_shown']}, noise {r['noise_shown']}, "
            f"silent when expected {r['silence_ok']}, "
            f"p50 {m['latency']['p50_ms']} ms, p95 {m['latency']['p95_ms']} ms"
        )
        u = r["use_detection"]
        g = m["graduation"]
        lines.append(
            f"   use detection: tp {u['tp']} fn {u['fn']} fp {u['fp']} | graduations "
            f"{g['ok']} ok / {g['failed']} failed, {g['model_calls']} model calls | "
            f"store {m['store']}"
        )
        if self.home:
            lines.append(f"   kept: {self.home}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "model": self.model,
            "embedder": self.embedder,
            "recall_mode": self.recall_mode,
            "ok": self.ok,
            "metrics": self.metrics,
            "outcomes": [o.__dict__ for o in self.outcomes],
            "items": self.items,
        }


@dataclass
class SimSession:
    name: str
    sid: str
    project: str | None
    cwd: Path
    transcript: Path
    last: str | None = None

    @property
    def key(self) -> str:
        return f"{AGENT}:{self.sid}"

    def append(self, type_: str, content: Any, **kw: Any) -> str:
        u = uuid.uuid4().hex
        role = "assistant" if type_ == "assistant" else "user"
        entry = {
            "uuid": u,
            "parentUuid": self.last,
            "type": type_,
            "isSidechain": False,
            "sessionId": self.sid,
            "cwd": str(self.cwd),
            "timestamp": clock.now().isoformat(),
            "message": {"role": role, "content": content},
            **kw,
        }
        with self.transcript.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.last = u
        return u


_REAL_EMBEDDER: Embedder | None = None


def real_embedder() -> Embedder:
    global _REAL_EMBEDDER
    if _REAL_EMBEDDER is None:
        from ..embedding import default_embedder

        _REAL_EMBEDDER = default_embedder()
        _REAL_EMBEDDER("warm up")
    return _REAL_EMBEDDER


class Simulator:
    def __init__(
        self,
        scenario: Scenario,
        *,
        model: str | Callable[[str], str] = "oracle",
        embedder: str = "hash",
        recall_mode: str = "on",
        keep: bool = False,
        log: Callable[[str], None] | None = None,
        settle_timeout: float | None = None,
    ) -> None:
        self.scenario = scenario
        self.recall_mode = recall_mode
        self.keep = keep
        self.log = log or (lambda _msg: None)
        self.oracle: OracleModel | None = None
        self.model_calls: list[dict[str, Any]] = []
        self.model_prompts: list[str] = []  # what the daemon sent the model, for checks
        if model == "oracle":
            self.oracle = OracleModel(scenario)
            self.ask: Callable[[str], str] = self.oracle
            self.model_name = "oracle"
        elif callable(model):
            self.ask, self.model_name = model, getattr(model, "__name__", "custom")
        else:
            from ..models import claude_code_model

            self.model_name = model
            self.ask = claude_code_model(model, 300, tempfile.gettempdir()).complete
        self.embedder_name = embedder
        self.embed: Embedder = hash_embed if embedder == "hash" else real_embedder()
        self.settle_timeout = settle_timeout or (60.0 if self.oracle else 1200.0)
        self.sessions: dict[str, SimSession] = {}
        self.outcomes: list[Outcome] = []
        self.batches: list[str] = []  # deletions a scenario made, newest last
        self.tally = Tally()
        self.events: list[dict[str, Any]] = []
        self._events_pos = 0
        self.daemon: Daemon | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ setup

    def _setup(self) -> None:
        # short: the socket path must stay under ~100 chars
        self.home = Path(tempfile.mkdtemp(prefix="gmsim"))
        self.work = self.home / "w"
        self.repos = {}
        for name in self.scenario.projects:
            root = self.work / name
            (root / ".git").mkdir(parents=True)
            self.repos[name] = root
        self.scratch = self.home / "scratch"
        self.scratch.mkdir()
        (self.home / "transcripts").mkdir()
        # a daemon autostarted by a hook by mistake must not touch real models
        (self.home / "config.toml").write_text(
            'model_backend = "none"\nembedder = "none"\ngraduation = false\n'
        )
        self._env = {
            k: os.environ.get(k)
            for k in ("GM_HOME", "GM_SOCKET", "GM_DISABLE", "CLAUDE_CODE_ENTRYPOINT", "GM_UI_PORT")
        }
        os.environ["GM_HOME"] = str(self.home)
        os.environ["GM_SOCKET"] = str(self.home / "gm.sock")
        os.environ.pop("GM_DISABLE", None)
        os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)  # a headless run skips the hooks
        os.environ["GM_UI_PORT"] = "0"  # its daemons serve no web page
        self.cfg: Config = replace(
            load_config(self.home),
            knowledge_recall=self.recall_mode,
            # sampled judging off: the oracle judges only what a scenario scripts, so it
            # runs just for notes that could have gone stale (always judged)
            judge_rate=0.0,
            events_days=0.0,  # simulated days pass quickly; the run reads the log as it grows
            save_interval=0.2,
            idle_timeout=10**7,
        )
        # held for the whole run, so a stray `gm daemon` exits instead of taking over
        self._lock = open(self.cfg.lock_path, "w")  # noqa: SIM115
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock.reset()

    def _teardown(self) -> None:
        with contextlib.suppress(Exception):
            self._stop_daemon()
        clock.reset()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._lock.close()
        if not self.keep:
            shutil.rmtree(self.home, ignore_errors=True)

    def _start_daemon(self) -> None:
        import asyncio

        cfg = self.cfg
        self.daemon = Daemon(
            cfg,
            knowledge=KnowledgeStore(cfg.home, self.embed),
            embedder=self.embed,
            graduation_model=self._ask,
            events=EventLog(events_path(cfg.home)),
        )
        loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=loop.run_until_complete, args=(self.daemon.serve(),), daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self.daemon.ready.is_set():
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("daemon did not start")
            time.sleep(0.01)

    def _stop_daemon(self) -> None:
        if self._thread is None:
            return
        with contextlib.suppress(DaemonUnavailable):
            call(self.cfg, "shutdown", autostart=False)
        self._thread.join(15)
        if self._thread.is_alive():
            raise RuntimeError("daemon did not stop")
        self._thread = None

    def _ask(self, prompt: str) -> str:
        started = time.perf_counter()
        kind = kind_of(prompt)
        self.model_prompts.append(prompt)
        try:
            return self.ask(prompt)
        finally:
            self.model_calls.append(
                {"kind": kind, "ms": round((time.perf_counter() - started) * 1000, 1)}
            )

    # ------------------------------------------------------------------ plumbing

    def _hook(self, cmd: str, payload: dict[str, Any]) -> str:
        old_in, old_out = sys.stdin, sys.stdout
        buf = io.StringIO()
        sys.stdin = io.StringIO(json.dumps(payload))
        sys.stdout = buf
        try:
            cli.main([cmd])
        finally:
            sys.stdin, sys.stdout = old_in, old_out
        return buf.getvalue()

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return call(self.cfg, method, params or {}, autostart=False)

    def settle(self) -> None:
        """Wait until the daemon has no background work left."""
        deadline = time.monotonic() + self.settle_timeout
        while not self._call("settled")["settled"]:
            if time.monotonic() > deadline:
                raise TimeoutError("daemon did not settle")
            time.sleep(0.02)

    def _new_events(self) -> list[dict[str, Any]]:
        path = events_path(self.cfg.home)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            fh.seek(self._events_pos)
            chunk = fh.read()
            self._events_pos = fh.tell()
        out = [json.loads(x) for x in chunk.splitlines() if x.strip()]
        self.events.extend(out)
        return out

    @staticmethod
    def _hook_errors(events: list[dict[str, Any]]) -> list[str]:
        return [
            f"hook {e.get('hook')} failed: {e.get('error')}"
            for e in events
            if e["kind"] == "hook" and not e.get("ok")
        ]

    def session(self, name: str) -> SimSession:
        if name not in self.sessions:
            project = self.scenario.sessions.get(name)
            cwd = self.repos[project] if project else self.scratch
            sid = f"{name}-{uuid.uuid4().hex[:12]}"
            self.sessions[name] = SimSession(
                name, sid, project, cwd, self.home / "transcripts" / f"{sid}.jsonl"
            )
        return self.sessions[name]

    # ------------------------------------------------------------------ state, for checks

    @property
    def store(self) -> KnowledgeStore:
        assert self.daemon is not None
        return self.daemon.knowledge

    def keys_of(self, text: str) -> set[str]:
        """The scenario facts a statement states."""
        if self.oracle is not None:
            key = self.oracle.key_of(text)
            return {key} if key else set()
        return {f.key for f in self.scenario.facts.values() if f.matches(text)}

    def items(self, key: str, valid_only: bool = False) -> list[Item]:
        out = [
            i
            for i in self.store.scan()
            if key in self.keys_of(i.statement) and (i.valid or not valid_only)
        ]
        return sorted(out, key=lambda i: i.created_at)

    def exchanges(self) -> list[dict[str, Any]]:
        """Every exchange stored so far (a turn that went on counts once)."""
        return [e for e in self.events_of("remember") if not e.get("continued")]

    def buffered(self) -> dict[str, int]:
        """Session -> exchanges still waiting in the buffer."""
        assert self.daemon is not None
        buf = self.daemon.buffer
        return {s: len(buf.exchanges(s)) for s in buf.sessions()}

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        self._new_events()
        return [e for e in self.events if e["kind"] == kind]

    # ------------------------------------------------------------------ run

    def run(self) -> Report:
        self._setup()
        try:
            self._start_daemon()
            self._new_events()
            for n, step in enumerate(self.scenario.steps, 1):
                try:
                    out = self._step(n, step)
                except Exception as exc:  # a broken step fails, the run carries on
                    out = Outcome(n, type(step).__name__.lower(), step.label, False, repr(exc))
                self.outcomes.append(out)
                self.log(
                    f"{'ok  ' if out.ok else 'FAIL'} #{n} {out.kind} {out.label} {out.detail}"[:200]
                )
            self.settle()
            if self.daemon is not None:  # the recall index missed no change
                diff = self.daemon.kindex.differences()
                self.outcomes.append(
                    Outcome(
                        len(self.outcomes) + 1,
                        "assert",
                        "the recall index matches a rebuild",
                        not diff,
                        ", ".join(diff)[:300],
                    )
                )
                # ... and what is on disk is what the daemon holds: no change went unsaved
                from ..knowledge import KnowledgeStore

                self.daemon.knowledge.save()
                disk = KnowledgeStore(self.cfg.home)
                try:
                    diff = self.daemon.knowledge.differences(disk)
                finally:
                    disk.close()
                self.outcomes.append(
                    Outcome(
                        len(self.outcomes) + 1,
                        "assert",
                        "memory on disk matches the daemon's",
                        not diff,
                        ", ".join(diff)[:300],
                    )
                )
            return self._report()
        finally:
            self._teardown()

    def _step(self, n: int, step: Step) -> Outcome:
        if isinstance(step, Turn):
            return self._turn(n, step)
        if isinstance(step, Compact):
            return self._compact(n, step)
        if isinstance(step, End):
            return self._end(n, step)
        if isinstance(step, Advance):
            clock.advance(seconds=step.seconds)
            got = self._call("maintain")
            self.settle()
            errors = self._hook_errors(self._new_events())
            days = step.seconds / 86400
            when = f"{days:g} days" if days >= 1 else f"{step.seconds / 60:g} min"
            return Outcome(
                n, "advance", step.label, not errors, f"+{when}: {got}; " + "; ".join(errors)
            )
        if isinstance(step, Restart):
            self._stop_daemon()
            if step.downtime:
                clock.advance(seconds=step.downtime)
            self._start_daemon()  # graduates sessions left idle while it was down
            self.settle()
            errors = self._hook_errors(self._new_events())
            st = self.store.stats()
            return Outcome(
                n,
                "restart",
                step.label,
                not errors,
                "; ".join(errors)
                or f"after {step.downtime / 3600:g}h down: {st['items']} items, "
                f"{st['segments']} segments",
            )
        if isinstance(step, Probe):
            return self._probe(n, step)
        if isinstance(step, Check):
            return self._check(n, step)
        if isinstance(step, Assert):
            problem = step.fn(self)
            return Outcome(n, "assert", step.label or step.name, problem is None, problem or "")
        if isinstance(step, Retract):
            return self._retract(n, step)
        if isinstance(step, FailNext):
            if self.oracle is None:
                return Outcome(n, "fail_next", step.label, True, "skipped: real model")
            self.oracle.fail_extract += step.extract
            self.oracle.fail_merge += step.merge
            self.oracle.garbage += step.garbage
            return Outcome(n, "fail_next", step.label, True, f"{step}")
        raise TypeError(step)

    # ------------------------------------------------------------------ steps

    def _retract(self, n: int, step: Retract) -> Outcome:
        """Through the real CLI, as a person would type it."""
        kind = "restore" if step.restore else "retract"
        cands = [i for i in self.items(step.key) if i.valid != step.restore]
        if not cands:
            return Outcome(n, kind, step.label, False, f"no {step.key!r} item to {kind}")
        it = cands[-1]
        argv = ["knowledge", kind, it.item_id[:10]]
        if step.reason:
            argv += ["--reason", step.reason]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        self._new_events()
        now = self.store.item(it.item_id)
        ok = code == 0 and now is not None and now.valid == step.restore
        return Outcome(n, kind, step.label, ok, (out.getvalue() + err.getvalue()).strip())

    def _recall(self, ses: SimSession, prompt: str) -> tuple[list[str], dict[str, Any] | None]:
        payload = {
            "session_id": ses.sid,
            "transcript_path": str(ses.transcript),
            "cwd": str(ses.cwd),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
        out = self._hook("recall", payload)
        events = self._new_events()
        ev = next((e for e in reversed(events) if e["kind"] == "knowledge_recall"), None)
        errors = self._hook_errors(events)
        if errors:
            raise RuntimeError("; ".join(errors))
        lines: list[str] = []
        if KNOWLEDGE_HEADER in out:
            block = out.split(KNOWLEDGE_HEADER, 1)[1].split("</graph-memory>", 1)[0]
            lines = [ln[2:] for ln in block.splitlines() if ln.startswith("- ")]
        if self.recall_mode != "on" and ev is not None:  # shadow: judge what it would show
            lines = [
                f"[{i.type.value}] {i.statement}" for i in self.store.get_items(ev.get("items", []))
            ]
        return lines, ev

    def _judge(self, lines: list[str], expect: Expect) -> tuple[bool, str, dict[str, Any]]:
        shown = []
        for ln in lines:
            body = note_statement(ln)
            shown.append((body, self.keys_of(body)))
        keys = set().union(*(k for _, k in shown)) if shown else set()
        if not expect.active:
            return True, f"shown {sorted(keys) or len(lines)}", {"shown": sorted(keys)}
        t = self.tally
        t.checks += 1
        problems = []
        missing = [k for k in expect.has if k not in keys]
        bad = [k for k in expect.lacks if k in keys]
        known = set(expect.has) | set(expect.allow) | set(expect.lacks)
        extra = [body for body, k in shown if not (k & known)]
        t.hits += len(expect.has) - len(missing)
        t.misses += len(missing)
        t.forbidden += len(bad)
        t.extras += len(extra)
        t.shown += len(shown)
        if expect.nothing:  # nothing beyond what `allow` permits
            if [body for body, k in shown if not (k & set(expect.allow))]:
                t.empty_violations += 1
                problems.append(f"expected nothing, shown {len(shown)}")
            else:
                t.empty_ok += 1
        if missing:
            problems.append(f"missing {missing}")
        if bad:
            problems.append(f"shown but should not be {bad}")
        data = {"shown": sorted(keys), "extra": extra, "lines": lines}
        detail = "; ".join(problems) or f"shown {sorted(keys)}" + (
            f" +{len(extra)} other" if extra else ""
        )
        return not problems, detail, data

    def _turn(self, n: int, st: Turn) -> Outcome:
        ses = self.session(st.session)
        clock.advance(seconds=self.scenario.turn_gap)
        lines, ev = self._recall(ses, st.prompt)
        ses.append("user", st.prompt)
        for i, f in enumerate(st.files):
            path = str(ses.cwd / f)
            tid = f"toolu_{uuid.uuid4().hex[:10]}"
            tool = "Edit" if i == 0 else "Read"
            inp: dict[str, Any] = {"file_path": path}
            if tool == "Edit":
                inp.update(old_string="a", new_string="b")
            ses.append("assistant", [{"type": "tool_use", "id": tid, "name": tool, "input": inp}])
            ses.append("user", [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}])
        for url in st.web:
            tid = f"toolu_{uuid.uuid4().hex[:10]}"
            ses.append(
                "assistant",
                [{"type": "tool_use", "id": tid, "name": "WebFetch", "input": {"url": url}}],
            )
            ses.append("user", [{"type": "tool_result", "tool_use_id": tid, "content": "page"}])
        ses.append("assistant", [{"type": "text", "text": st.reply}])
        self._hook(
            "remember",
            {
                "session_id": ses.sid,
                "transcript_path": str(ses.transcript),
                "cwd": str(ses.cwd),
                "hook_event_name": "Stop",
            },
        )
        self.settle()
        events = self._new_events()
        errors = self._hook_errors(events)
        ok, detail, data = self._judge(lines, st.expect)
        problems = [] if ok else [detail]
        problems += errors
        if ev is not None:
            data["recall_reason"] = ev.get("reason")
            data["recall_ms"] = ev.get("ms")
        if st.used or st.unused:
            use = next(
                (
                    e
                    for e in reversed(events)
                    if e["kind"] == "knowledge_use" and e["session"] == ses.key
                ),
                None,
            )
            used = set()
            if use is not None:
                for it in self.store.get_items(use.get("used", [])):
                    used |= self.keys_of(it.statement)
            for k in st.used:
                if k in used:
                    self.tally.use_tp += 1
                else:
                    self.tally.use_fn += 1
                    problems.append(f"reply used {k} but it was not counted")
            for k in st.unused:
                if k in used:
                    self.tally.use_fp += 1
                    problems.append(f"{k} counted as used but the reply ignored it")
            data["used"] = sorted(used)
        detail = "; ".join(problems) if problems else detail
        return Outcome(
            n, "turn", st.label or f"{st.session}: {st.prompt[:50]}", not problems, detail, data
        )

    def _graduate_hook(
        self, ses: SimSession, payload: dict[str, Any], fails: bool = False
    ) -> list[str]:
        self._hook("graduate", payload)
        self.settle()
        events = self._new_events()
        errors = self._hook_errors(events)
        failed = [e for e in events if e["kind"] == "graduate" and not e.get("ok")]
        if fails and not failed:
            errors.append("graduation was expected to fail but did not")
        if not fails:
            errors += [f"graduation failed: {e.get('error')}" for e in failed]
        return errors

    def _grad_detail(self) -> str:
        grads = [e for e in self.events if e["kind"] == "graduate"]
        if not grads:
            return "no graduation"
        g = grads[-1]
        return (
            f"graduated {g.get('exchanges')} exchanges -> {g.get('items')} items "
            f"({g.get('new')} new)"
        )

    def _compact(self, n: int, st: Compact) -> Outcome:
        ses = self.session(st.session)
        ses.last = None  # Claude Code starts a new chain at the compact boundary
        ses.append("system", "Conversation compacted", subtype="compact_boundary")
        ses.append("user", st.summary, isCompactSummary=True, isVisibleInTranscriptOnly=True)
        before = len(self.events_of("graduate"))
        errors = self._graduate_hook(
            ses,
            {
                "session_id": ses.sid,
                "transcript_path": str(ses.transcript),
                "cwd": str(ses.cwd),
                "hook_event_name": "SessionStart",
                "source": "compact",
            },
            st.fails,
        )
        after = len(self.events_of("graduate"))
        detail = self._grad_detail() if after > before else "nothing to graduate"
        return Outcome(
            n, "compact", st.label or st.session, not errors, "; ".join(errors) or detail
        )

    def _end(self, n: int, st: End) -> Outcome:
        ses = self.session(st.session)
        before = len(self.events_of("graduate"))
        errors = self._graduate_hook(
            ses,
            {
                "session_id": ses.sid,
                "transcript_path": str(ses.transcript),
                "cwd": str(ses.cwd),
                "hook_event_name": "SessionEnd",
                "reason": "prompt_input_exit",
            },
            st.fails,
        )
        after = len(self.events_of("graduate"))
        detail = self._grad_detail() if after > before else "nothing to graduate"
        return Outcome(n, "end", st.label or st.session, not errors, "; ".join(errors) or detail)

    def _probe(self, n: int, st: Probe) -> Outcome:
        name = f"probe{n}"
        self.scenario.sessions[name] = st.project
        ses = self.session(name)
        lines, ev = self._recall(ses, st.prompt)
        ok, detail, data = self._judge(lines, st.expect)
        if ev is not None:
            data["recall_reason"] = ev.get("reason")
            data["why"] = ev.get("why")
        return Outcome(n, "probe", st.label or st.prompt[:60], ok, detail, data)

    def _check(self, n: int, st: Check) -> Outcome:
        items = self.items(st.key)
        problems = []
        if st.count is not None and len(items) != st.count:
            problems.append(f"{len(items)} items state it, expected {st.count}")
        if st.min_count is not None and len(items) < st.min_count:
            problems.append(f"{len(items)} items state it, expected at least {st.min_count}")
        valid = [i for i in items if i.valid]
        if st.valid is True and not valid:
            problems.append("no valid item states it")
        if st.valid is False and valid:
            problems.append("still valid")
        # the first item stating it; a real model may later learn related knowledge
        cur = valid[0] if valid else (items[-1] if items else None)
        if cur is None:
            if st.count != 0:
                problems.append("no item states it")
        else:
            v = cur.vitality(clock.now())
            if st.area is not None and cur.area.value != st.area:
                problems.append(f"area {cur.area.value}, expected {st.area}")
            if st.min_strength is not None and cur.strength < st.min_strength:
                problems.append(f"strength {cur.strength} < {st.min_strength}")
            if st.max_strength is not None and cur.strength > st.max_strength:
                problems.append(f"strength {cur.strength} > {st.max_strength}")
            if st.min_uses is not None and cur.uses < st.min_uses:
                problems.append(f"uses {cur.uses} < {st.min_uses}")
            state = (
                f"{len(items)} item(s); {cur.area.value}, strength {cur.strength:g}, "
                f"vitality {v:.2f}, uses {cur.uses}, valid {cur.valid}"
            )
        detail = "; ".join(problems) or (state if cur else "absent, as expected")
        return Outcome(n, "check", st.label or st.key, not problems, detail)

    # ------------------------------------------------------------------ report

    def _report(self) -> Report:
        self._new_events()
        recalls = [e["ms"] for e in self.events if e["kind"] == "knowledge_recall" and "ms" in e]
        recalls.sort()

        def pct(p: float) -> float | None:
            return recalls[min(len(recalls) - 1, int(p * len(recalls)))] if recalls else None

        grads = [e for e in self.events if e["kind"] == "graduate"]
        st = self.store.stats()
        metrics = {
            "recall": self.tally.summary(),
            "latency": {"recalls": len(recalls), "p50_ms": pct(0.5), "p95_ms": pct(0.95)},
            "graduation": {
                "ok": sum(1 for g in grads if g.get("ok")),
                "failed": sum(1 for g in grads if not g.get("ok")),
                "model_calls": len(self.model_calls),
                "model_ms": round(sum(c["ms"] for c in self.model_calls)),
                "triggers": sorted({g.get("trigger", "?") for g in grads}),
            },
            "store": {
                "items": st["items"],
                "invalid": st["invalidated"],
                "entities": st["entities"],
                "edges": st["edges"],
                "areas": st.get("by_area"),
            },
        }
        return Report(
            self.scenario.name,
            self.scenario.description,
            self.model_name,
            self.embedder_name,
            self.recall_mode,
            self.outcomes,
            metrics,
            str(self.home) if self.keep else None,
            [
                {
                    "facts": sorted(self.keys_of(i.statement)),
                    "type": i.type.value,
                    "statement": i.statement,
                    "scope": i.scope.rsplit("/", 1)[-1],
                    "area": i.area.value,
                    "valid": i.valid,
                    "strength": i.strength,
                    "uses": i.uses,
                }
                for i in sorted(self.store.scan(), key=lambda i: i.created_at)
            ],
        )


SimState = Simulator


def run(scenario: Scenario, **kw: Any) -> Report:
    return Simulator(scenario, **kw).run()
