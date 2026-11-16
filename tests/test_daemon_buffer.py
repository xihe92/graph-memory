"""The daemon and the session buffer: what is kept, for how long, and what survives
failures."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graph_memory import clock
from graph_memory.buffer import Exchange, Summary
from graph_memory.client import DaemonUnavailable, RemoteError, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon

from .conftest import turn
from .test_daemon import _start, _wait
from .test_graduation import LOCK, Script


@pytest.fixture
def cfg(short_dir: Path) -> Config:
    return replace(load_config(short_dir), save_interval=0.05)


@contextlib.contextmanager
def daemon_with(cfg: Config, ask: Callable[[str], str] | None) -> Iterator[Daemon]:
    daemon, t = _start(cfg, ask)
    try:
        yield daemon
    finally:
        with contextlib.suppress(DaemonUnavailable, OSError):
            call(cfg, "shutdown", autostart=False)
        t.join(5)
        clock.reset()


def _remember(cfg: Config, session: str, prompt: str, reply: str, **kw: Any) -> dict[str, Any]:
    return call(  # type: ignore[no-any-return]
        cfg, "remember", {"session": session, "turn": turn(prompt, reply, **kw)}, autostart=False
    )


def _graduate(cfg: Config, session: str, trigger: str, summary: str = "") -> None:
    call(
        cfg,
        "graduate",
        {"session": session, "trigger": trigger, "compact_summary": summary},
        autostart=False,
    )


def _events(cfg: Config, kind: str) -> list[dict[str, Any]]:
    lines = (cfg.home / "events.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["kind"] == kind]


def test_rewind_continued_turn_and_forget(cfg: Config) -> None:
    with daemon_with(cfg, None) as d:
        for n in range(3):
            _remember(cfg, "s", f"q{n}", f"a{n}", source={"first": f"u{n}", "last": f"a{n}"})
        # the user rewound to after the first reply and asked something else
        res = _remember(cfg, "s", "q-new", "a-new", previous_reply_ref="a0", source={"first": "u9"})
        assert res["rewound"] == 2
        assert [x.prompt for x in d.buffer.exchanges("s")] == ["q0", "q-new"]
        assert _events(cfg, "rewind")[0]["dropped"]
        # a reply after a background task finished: the same prompt, a longer reply
        res = _remember(cfg, "s", "q-new", "a-new, then the tests passed", source={"first": "u9"})
        assert res["continued"] and res["exchange"] == d.buffer.exchanges("s")[1].id
        assert d.buffer.exchanges("s")[1].reply == "a-new, then the tests passed"
        assert len(d.buffer.exchanges("s")) == 2
        # an unknown earlier reply is not a rewind
        assert _remember(cfg, "s", "q3", "a3", previous_reply_ref="zzz")["rewound"] == 0
        first = d.buffer.exchanges("s")[0].id
        assert call(cfg, "forget", {"session": "s", "id": first}, autostart=False)
        with pytest.raises(RemoteError, match="no exchange"):
            call(cfg, "forget", {"session": "s", "id": first}, autostart=False)
        assert [x.prompt for x in d.buffer.exchanges("s")] == ["q-new", "q3"]


def test_later_graduations_read_the_compaction_summary(cfg: Config) -> None:
    ask = Script([LOCK, {"entities": [], "items": []}])
    with daemon_with(cfg, ask) as d:
        _remember(cfg, "s", "add a lock", "done")
        _graduate(cfg, "s", "compact", "SUMMARY: we added a SETNX lock to the limiter")
        _wait(lambda: d.knowledge.stats()["segments"] == 1)
        _wait(lambda: [r.kind for r in d.buffer.records("s")] == ["summary"])
        (summary,) = d.buffer.summaries("s")
        assert summary.text.startswith("SUMMARY:") and summary.trigger == "compact"
        _remember(cfg, "s", "and retries?", "added")
        _graduate(cfg, "s", "session_end")
        _wait(lambda: d.buffer.sessions() == [])
        extracts = [p for p in ask.prompts if "distil" in p]
        assert len(extracts) == 2 and "SUMMARY: we added" in extracts[1]
        cleared = _events(cfg, "graduate")[-1]
        assert cleared["ok"] and cleared["cleared"]


def test_a_slow_session_keeps_one_growing_summary(cfg: Config) -> None:
    """A question, a long pause, another question: each pause graduates what came before.
    Each graduation reads the summary the previous one wrote and writes the next; one that
    learns nothing new folds into the summary before it."""
    nothing = {"entities": [], "items": []}
    ask = Script(
        [
            {**LOCK, "summary": "Added a SETNX lock to the limiter."},
            {**nothing, "summary": "Added a SETNX lock; then asked about the expiry."},
            {**nothing, "summary": "Added a SETNX lock; expiry and retries discussed."},
            {**nothing},  # a model that wrote no summary: the last one stays
        ]
    )
    with daemon_with(cfg, ask) as d:
        for n, (q, a) in enumerate(
            [("add a lock", "done"), ("why 30s?", "jobs take 10s"), ("retries?", "5"), ("ok", "ok")]
        ):
            _remember(cfg, "s", q, a)
            _graduate(cfg, "s", "idle")
            _wait(lambda n=n: d.knowledge.stats()["segments"] == n + 1)  # type: ignore[misc]
            _wait(lambda: d.buffer.exchanges("s") == [])
        (summary,) = d.buffer.records("s")
        assert isinstance(summary, Summary)
        assert summary.text == "Added a SETNX lock; expiry and retries discussed."
        assert summary.exchanges == 4 and summary.graduations == 4
        assert len(summary.items) == 2  # the two the first graduation learned
        extracts = [p for p in ask.prompts if "distil" in p]
        assert "<earlier>" not in extracts[0]
        assert "<earlier>\nAdded a SETNX lock to the limiter.\n</earlier>" in extracts[1]
        assert "expiry and retries discussed.\n</earlier>" in extracts[3]
        # the Sessions page: one summary node, no separate graduation nodes for the
        # graduations folded into it, and its items readable
        (view,) = call(cfg, "sessions", autostart=False)["sessions"]
        assert [r["kind"] for r in view["records"]] == ["summary"]
        assert {i["type"] for i in view["records"][0]["items"]} == {"decision", "procedure"}


def test_only_a_compaction_starts_a_new_summary(cfg: Config) -> None:
    nothing = {"entities": [], "items": []}
    ask = Script(
        [
            {**nothing, "summary": "talked"},
            {**LOCK, "summary": "talked, then a lock"},
            nothing,
            {**nothing, "summary": "after the compaction"},
            {**nothing, "summary": "after the compaction, more"},
        ]
    )
    with daemon_with(cfg, ask) as d:

        def step(prompt: str, trigger: str, summaries: int, compact: str = "") -> None:
            n = d.knowledge.stats()["segments"]
            _remember(cfg, "s", prompt, "ok")
            _graduate(cfg, "s", trigger, compact)
            _wait(lambda: d.knowledge.stats()["segments"] == n + 1)
            _wait(lambda: d.buffer.exchanges("s") == [])
            assert len(d.buffer.summaries("s")) == summaries

        step("hello", "idle", 1)
        step("add a lock", "idle", 1)  # learned something: still the same summary
        step("more", "compact", 2, "SUMMARY: the compaction")  # a compaction: its own
        step("then", "idle", 3)  # the next rolling summary starts after it
        step("and then", "idle", 3)
        assert [(x.text, x.graduations) for x in d.buffer.summaries("s")] == [
            ("talked, then a lock", 2),
            ("SUMMARY: the compaction", 1),
            ("after the compaction, more", 2),
        ]
        assert len(d.buffer.summaries("s")[0].items) == 2  # what the lock graduation learned


def test_a_crash_between_saving_knowledge_and_the_buffer_learns_nothing_twice(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    with daemon_with(cfg, Script([LOCK, LOCK])) as d:
        real = d.buffer.summarize
        calls = {"n": 0}

        def crash_once(*a: Any, **kw: Any) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            return real(*a, **kw)

        monkeypatch.setattr(d.buffer, "summarize", crash_once)
        _remember(cfg, "s", "add a lock", "done")
        _graduate(cfg, "s", "idle")
        _wait(lambda: d.knowledge.stats()["segments"] == 1)
        _wait(d.settled)
        assert len(d.buffer.exchanges("s")) == 1  # still waiting: the rewrite crashed
        items = d.knowledge.stats()["items"]
        _graduate(cfg, "s", "idle")  # graduated again
        _wait(lambda: not d.buffer.exchanges("s"))
        assert d.knowledge.stats()["items"] == items  # the same statements: reinforced
        assert all(i.strength == 2.0 for i in d.knowledge.scan())


def test_without_a_model_nothing_outlives_the_session(cfg: Config) -> None:
    with daemon_with(cfg, None) as d:
        _remember(cfg, "s", "hello", "hi")
        _graduate(cfg, "s", "idle")
        _wait(d.settled)
        assert len(d.buffer.exchanges("s")) == 1  # nothing to learn with, nothing done
        _graduate(cfg, "s", "session_end")
        _wait(lambda: d.buffer.sessions() == [])
        assert _events(cfg, "session_cleared")[0]["trigger"] == "session_end"


def test_abandoned_sessions_graduate_and_go(cfg: Config) -> None:
    with daemon_with(cfg, Script([LOCK])) as d:
        _remember(cfg, "s", "add a lock", "done")
        clock.advance(days=8)
        assert call(cfg, "maintain", autostart=False)["queued"] == 1
        _wait(lambda: d.buffer.sessions() == [])
        (seg,) = d.knowledge.segments()
        assert seg.trigger == "abandoned"
        # a folder left holding only a summary goes too
        x = d.buffer.append("t", Exchange(prompt="x", reply="y"))
        d.buffer.summarize("t", [x.id], Summary(trigger="idle"))
        clock.advance(days=8)
        call(cfg, "maintain", autostart=False)
        _wait(lambda: d.buffer.sessions() == [])


def test_a_graduation_gives_up_once_the_exchanges_are_too_old(cfg: Config) -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("model down")

    with daemon_with(replace(cfg, graduation_attempts=10), broken) as d:
        _remember(cfg, "s", "add a lock", "done")
        _graduate(cfg, "s", "idle")
        _wait(lambda: len(_events(cfg, "graduate")) == 1)
        assert d.buffer.exchanges("s")  # one failure: kept for the next try
        clock.advance(days=4)
        _graduate(cfg, "s", "idle")
        _wait(lambda: not d.buffer.exchanges("s"))
        fail = _events(cfg, "graduate")[-1]
        assert not fail["ok"] and fail["given_up"] and fail["attempt"] == 2
        assert _events(cfg, "buffer_dropped")[0]["exchanges"] == 1


def test_an_exchange_stored_during_graduation_waits_for_the_next(cfg: Config) -> None:
    gate = threading.Event()
    script = Script([LOCK])

    def slow(prompt: str) -> str:
        gate.wait(5)
        return script(prompt)

    with daemon_with(cfg, slow) as d:
        _remember(cfg, "s", "add a lock", "done")
        _graduate(cfg, "s", "session_end")
        _wait(lambda: d._graduating == "s")
        _remember(cfg, "s", "one more thing", "sure")  # after the session-end hook
        gate.set()
        _wait(lambda: d.knowledge.stats()["segments"] == 1)
        _wait(d.settled)
        recs = d.buffer.records("s")
        assert [r.kind for r in recs] == ["summary", "exchange"]
        assert recs[1].prompt == "one more thing"  # type: ignore[union-attr]


def test_backfill_queue_dedup_and_leftover_folders(cfg: Config) -> None:
    gate = threading.Event()
    script = Script([LOCK, {"entities": [], "items": []}])

    def slow(prompt: str) -> str:
        gate.wait(5)
        return script(prompt)

    with daemon_with(cfg, slow) as d:
        _remember(cfg, "a", "add a lock", "done")
        _remember(cfg, "b", "hello", "hi")
        assert call(cfg, "graduate", {"all": True}, autostart=False) == {"queued": 2}
        _wait(lambda: d._graduating is not None)
        # one graduation already waits for b: an idle trigger adds nothing
        assert not d.enqueue_graduation("b", "idle")
        gate.set()
        _wait(lambda: d.knowledge.stats()["segments"] == 2)
        _wait(d.settled)
        # b now holds only a summary; its session ending deletes the folder
        _graduate(cfg, "b", "session_end")
        _wait(lambda: "b" not in d.buffer.sessions())
        # an empty folder (a crash right after creating it) is cleared by the sweep
        empty = d.buffer._path("c")
        empty.parent.mkdir(parents=True)
        empty.write_text("")
        call(cfg, "maintain", autostart=False)
        assert "c" not in d.buffer.sessions()
        assert d.buffer.stats()["summaries"] == 1  # a's, from the manual graduation


def test_a_live_session_shows_even_when_all_of_it_graduated(cfg: Config) -> None:
    with daemon_with(cfg, Script([LOCK])) as d:
        call(cfg, "recall", {"session": "s", "prompt": "add a lock"}, autostart=False)
        _remember(cfg, "s", "add a lock", "done")
        _graduate(cfg, "s", "manual")
        _wait(lambda: d.knowledge.stats()["segments"] == 1)
        _wait(d.settled)
        call(cfg, "recall", {"session": "s", "prompt": "now the retries"}, autostart=False)
        view = call(cfg, "sessions", autostart=False)
        (sess,) = view["sessions"]
        kinds = [r["kind"] for r in sess["records"]]
        # the graduation shows once (as the buffer's summary), then the prompt in progress
        assert kinds == ["summary", "asking"] and sess["waiting"] == 0
        assert sess["records"][1]["prompt"] == "now the retries"

        # the buffer emptied by hand: the graduation itself stands in for it
        d.buffer.delete("s")
        (sess,) = call(cfg, "sessions", autostart=False)["sessions"]
        graduated, asking = sess["records"]
        assert graduated["kind"] == "graduated" and graduated["exchanges"] == 1
        assert {i["statement"] for i in graduated["items"]} == {
            i.statement for i in d.knowledge.scan()
        }
        assert all(i["new"] and i["valid"] for i in graduated["items"])

        _graduate(cfg, "s", "session_end")  # ended: no longer live
        _wait(lambda: call(cfg, "sessions", autostart=False)["sessions"] == [])


def test_live_sessions_survive_a_restart(cfg: Config) -> None:
    with daemon_with(cfg, None):
        call(cfg, "recall", {"session": "a", "prompt": "hello", "cwd": "/"}, autostart=False)
        _remember(cfg, "b", "hi", "there")
        _graduate(cfg, "b", "session_end")
        call(cfg, "recall", {"session": "c", "prompt": "answered", "cwd": "/"}, autostart=False)
        _remember(cfg, "c", "answered", "yes")
        _wait(lambda: len(call(cfg, "sessions", autostart=False)["sessions"]) == 2)
    with daemon_with(cfg, None) as d:
        view = call(cfg, "sessions", autostart=False)
        assert sorted(s["session"] for s in view["sessions"]) == ["a", "c"]  # b ended
        # a's prompt never got its reply: still in progress after the restart; c's did
        assert set(d._asking) == {"a"} and d._asking["a"][0] == "hello"


def test_a_session_with_nothing_to_show_fades_after_half_an_hour(cfg: Config) -> None:
    with daemon_with(cfg, None) as d:
        call(cfg, "recall", {"session": "x", "prompt": "hi"}, autostart=False)
        d._asking.clear()  # say it never got a reply stored (a script's run)
        assert [s["session"] for s in call(cfg, "sessions", autostart=False)["sessions"]] == ["x"]
        clock.advance(minutes=31)
        assert call(cfg, "sessions", autostart=False)["sessions"] == []


def test_a_restart_does_not_show_a_session_its_notes_again(cfg: Config) -> None:
    """The daemon stops when idle. What a session was shown, and that it had its standing
    preferences, comes back from the event log; a compaction still starts both over."""
    from graph_memory.knowledge import Item, ItemType

    q = "how does the limiter acquire its redis lock?"

    def notes(session: str) -> list[str]:
        res = call(cfg, "recall", {"session": session, "prompt": q}, autostart=False)
        return list(res["context"].get("knowledge") or [])

    with daemon_with(cfg, None) as d:
        d.knowledge.add_item(
            Item(type=ItemType.FACT, statement="The limiter acquires its redis lock with SETNX")
        )
        d.knowledge.add_item(Item(type=ItemType.PREFERENCE, statement="Answer in plain terms"))
        d.knowledge.save()
        first = notes("s1")
        assert len(first) == 2  # the standing preference and the fact
        assert notes("s1") == []
    with daemon_with(cfg, None):
        assert notes("s1") == []  # shown before the restart
        assert len(notes("s2")) == 2  # another session has seen nothing
        _graduate(cfg, "s1", "compact", "summary")
        assert len(notes("s1")) == 2  # its context was compacted: both may come again
    with daemon_with(cfg, None):
        assert notes("s1") == []  # shown since that compaction
        _graduate(cfg, "s2", "session_end")
        while not call(cfg, "settled", autostart=False)["settled"]:
            time.sleep(0.01)
    with daemon_with(cfg, None) as d:
        assert "s2" not in d._shown and "s2" not in d._greeted  # ended: forgotten
