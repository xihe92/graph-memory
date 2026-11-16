"""Prompts about what is left get the open items; a model judges notes on sampled turns;
the event log keeps a window of days."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from graph_memory import clock
from graph_memory.client import DaemonUnavailable, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon
from graph_memory.events import EventLog, events_path, prune_events, read_events
from graph_memory.judge import build_judge_prompt, parse_judgement
from graph_memory.knowledge import Item, ItemType, KnowledgeStore
from graph_memory.knowledge_recall import (
    KnowledgeIndex,
    asks_about_remaining_work,
    project_named,
    recall,
)
from graph_memory.stats import format_help, memory_help

from .conftest import topic_embed, turn
from .test_daemon import _wait

GM = "repo:/Users/me/workspace/graph_memory"
IC = "repo:/Users/me/workspace/trip-planner"


def test_which_prompts_ask_about_remaining_work() -> None:
    for p in (
        "what's left?",
        "What is left to do on the limiter?",
        "anything still open here",
        "where were we?",
        "where did we leave off with this",
        "remaining tasks for graph memory",
        "continue",
        "what's next",
    ):
        assert asks_about_remaining_work(p), p
    for p in (
        "fix the redis race",
        "what is a SETNX lock",
        "left-pad the numbers",
        "open the file",
    ):
        assert not asks_about_remaining_work(p), p


def _store(tmp_path: Path) -> tuple[KnowledgeStore, dict[str, Item]]:
    k = KnowledgeStore(tmp_path, topic_embed)
    t0 = datetime(2026, 11, 1, tzinfo=UTC)
    items = {
        "gm_old": Item(type=ItemType.OPEN, statement="graph memory: write the migration", scope=GM),
        "gm_new": Item(type=ItemType.OPEN, statement="graph memory: judge notes live", scope=GM),
        "ic": Item(type=ItemType.OPEN, statement="trip-planner: test the login", scope=IC),
        "fact": Item(type=ItemType.FACT, statement="graph memory runs as a daemon", scope=GM),
    }
    for n, it in enumerate(items.values()):
        it.created_at = t0 + timedelta(hours=n)
        k.add_item(it)
    items["gm_old"].created_at = t0 - timedelta(days=1)
    return k, items


def test_open_items_follow_the_named_or_session_project(tmp_path: Path) -> None:
    k, items = _store(tmp_path)
    ix = KnowledgeIndex(k)
    assert project_named(k, "what's left on trip planner?") == IC
    assert project_named(k, "where were we with graph_memory") == GM
    assert project_named(k, "where were we") is None

    def ids(prompt: str, **kw: Any) -> list[str]:
        return [i.item_id for i in recall(k, ix, prompt, **kw).items]

    # named project wins over the session's
    assert ids("what's left for trip-planner?", project=GM)[0] == items["ic"].item_id
    # the session's project, newest first
    got = recall(k, ix, "where were we?", project=GM)
    assert [i.item_id for i in got.items][:2] == [items["gm_new"].item_id, items["gm_old"].item_id]
    assert got.why[items["gm_new"].item_id] == ["open item: the prompt asks what is left"]
    assert items["ic"].item_id not in [i.item_id for i in got.items]
    # nothing else matches "continue": the reason says the open items are all there is
    bare = KnowledgeStore(tmp_path / "bare", None)
    bare.add_item(Item(type=ItemType.OPEN, statement="trip-planner: test the login", scope=IC))
    only = recall(bare, KnowledgeIndex(bare), "continue", project=IC)
    assert only.reason == "open items only" and len(only.items) == 1
    # no project at all: the newest anywhere, at most three
    assert len(ids("what's next?")) == 3
    # already shown in this session: not again
    assert items["gm_new"].item_id not in ids(
        "where were we?", project=GM, exclude=[items["gm_new"].item_id]
    )
    # an ordinary prompt gets no open items it doesn't match
    assert items["ic"].item_id not in ids("graph memory daemon", project=GM)


@pytest.fixture
def cfg(short_dir: Path) -> Config:
    return replace(load_config(short_dir), save_interval=0.05)


@contextlib.contextmanager
def judged_daemon(
    cfg: Config, judge: Any, items: list[Item], sample: float = 0.0
) -> Iterator[Daemon]:
    """`sample`: what the daemon's dice roll; 0 samples every turn, 1 none."""
    import asyncio
    import threading

    k = KnowledgeStore(cfg.home, topic_embed)
    for it in items:
        k.add_item(it)
    k.save()
    d = Daemon(
        cfg,
        knowledge=KnowledgeStore(cfg.home, topic_embed),
        embedder=topic_embed,
        events=EventLog(events_path(cfg.home)),
        judge_model=judge,
        rng=lambda: sample,
    )
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_until_complete, args=(d.serve(),), daemon=True)
    t.start()
    _wait(d.ready.is_set)
    try:
        yield d
    finally:
        with contextlib.suppress(DaemonUnavailable, OSError):
            call(cfg, "shutdown", autostart=False)
        t.join(5)


def _events(cfg: Config, kind: str) -> list[dict[str, Any]]:
    return list(read_events(events_path(cfg.home), kinds=[kind]))


def test_cross_project_open_items_say_which_project(cfg: Config, tmp_path: Path) -> None:
    item = Item(type=ItemType.OPEN, statement="trip-planner: test the calendar login", scope=IC)
    with judged_daemon(cfg, None, [item]):
        res = call(
            cfg,
            "recall",
            {"session": "s", "prompt": "what's left?", "cwd": str(tmp_path)},
            autostart=False,
        )
        (line,) = res["context"]["knowledge"]
        day = item.created_at.date().isoformat()
        assert line == (
            f"[open, {day}] trip-planner: test the calendar login [project: trip-planner]"
        )


def test_sampled_turns_get_a_verdict_per_note(cfg: Config) -> None:
    lock = Item(type=ItemType.DECISION, statement="the limiter uses a redis lua script lock")
    rain = Item(type=ItemType.FACT, statement="redis retries back off on lua errors")
    prompts: list[str] = []

    def judge(prompt: str) -> str:
        prompts.append(prompt)
        return '```json\n{"A1": "useful", "A2": "misleading", "A9": "useful", "A1x": "x"}\n```'

    with judged_daemon(cfg, judge, [lock, rain]) as d:
        r = call(
            cfg, "recall", {"session": "s", "prompt": "redis lua lock retries"}, autostart=False
        )
        shown = r["context"]["knowledge"]
        assert len(shown) == 2
        call(
            cfg,
            "remember",
            {"session": "s", "turn": turn("redis lua lock retries", "lua")},
            autostart=False,
        )
        _wait(lambda: bool(_events(cfg, "knowledge_judge")))
        _wait(d.settled)
        (ev,) = _events(cfg, "knowledge_judge")
        assert ev["ok"] and set(ev["labels"].values()) == {"useful", "misleading"}
        assert "<reply>\nlua\n</reply>" in prompts[0] and "A2 [" in prompts[0]

    h = memory_help(read_events(events_path(cfg.home)), tz=UTC)
    j = h["judged"]
    assert (j["turns"], j["notes"], j["useful"], j["misleading"]) == (1, 2, 1, 1)
    assert j["useful_rate"] == 0.5 and j["most_misleading"][0]["times"] == 1
    assert h["turns"][0]["judged"] == ev["labels"]
    text = format_help(h)
    assert "judged         1 turns by a model: 1 of 2 notes useful (50%)" in text


def test_a_failed_verdict_is_logged_and_turns_off_without_a_model(cfg: Config) -> None:
    lock = Item(type=ItemType.DECISION, statement="the limiter uses a redis lua script lock")

    def broken(prompt: str) -> str:
        return "I think they are all fine."

    with judged_daemon(cfg, broken, [lock]) as d:
        call(cfg, "recall", {"session": "s", "prompt": "redis lua lock"}, autostart=False)
        call(
            cfg, "remember", {"session": "s", "turn": turn("redis lua lock", "ok")}, autostart=False
        )
        _wait(lambda: bool(_events(cfg, "knowledge_judge")))
        _wait(d.settled)
    (ev,) = _events(cfg, "knowledge_judge")
    assert not ev["ok"] and "no JSON" in ev["error"]
    assert memory_help(read_events(events_path(cfg.home)))["judged"]["failed"] == 1


def test_judge_prompt_clips_long_replies() -> None:
    it = Item(type=ItemType.FACT, statement="s", why="w")
    p = build_judge_prompt("q", "x" * 10000, [it])
    assert "[…]" in p and f"A1 [fact, {it.created_at.date()}] s (why: w)" in p
    with pytest.raises(ValueError):
        parse_judgement("no braces here", [it])
    assert parse_judgement('{"A1": "great", "B2": "useful"}', [it]) == {}


# --------------------------------------------------------------------------- event window


def test_prune_events_keeps_the_window(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    old = path.with_name("events.jsonl.1")
    now = datetime(2026, 11, 5, tzinfo=UTC)

    def line(days: float, kind: str = "hook") -> str:
        return json.dumps({"ts": (now - timedelta(days=days)).isoformat(), "kind": kind}) + "\n"

    assert prune_events(path, now) == 0  # no files: nothing to do
    old.write_text(line(40) + line(35))
    path.write_text(line(31) + "{broken\n" + line(10) + line(1))
    assert prune_events(path, now - timedelta(days=30)) == 4
    assert not path.exists() or path.read_text() == ""
    kept = list(read_events(path))
    assert len(kept) == 2
    before = old.read_text()
    assert prune_events(path, now - timedelta(days=30)) == 0  # nothing old: untouched
    assert old.read_text() == before


def test_the_daemon_prunes_once_a_day(cfg: Config) -> None:
    path = events_path(cfg.home)
    stale = (clock.now() - timedelta(days=60)).isoformat()
    path.write_text(json.dumps({"ts": stale, "kind": "hook"}) + "\n")
    d = Daemon(cfg, events=EventLog(path))
    assert d.prune_events() == 1 and d.prune_events() == 0  # again within the day: skipped
    path.write_text(json.dumps({"ts": stale, "kind": "hook"}) + "\n")
    assert Daemon(replace(cfg, events_days=0), events=EventLog(path)).prune_events() == 0
    assert Daemon(cfg, events=EventLog(None)).prune_events() == 0


def _turn(cfg: Config, session: str, prompt: str, reply: str) -> list[str]:
    shown = call(cfg, "recall", {"session": session, "prompt": prompt}, autostart=False)
    call(cfg, "remember", {"session": session, "turn": turn(prompt, reply)}, autostart=False)
    return list(shown["context"].get("knowledge") or [])


def test_a_contradicted_note_is_disputed_then_retracted(cfg: Config) -> None:
    from graph_memory.judge import verdicts
    from graph_memory.knowledge import Volatility

    ids = Item(
        type=ItemType.LEARNED,
        statement="sqlite foreign keys are on by default",
        volatility=Volatility.VERSIONED,
    )
    seen: list[str] = []

    def judge(prompt: str) -> str:
        seen.append(prompt)
        return '{"A1": {"label": "contradicted", "why": "the docs say they are off by default"}}'

    # never sampled: the note is versioned, so every turn that shows it is judged
    with judged_daemon(cfg, judge, [ids], sample=1.0) as d:
        assert len(_turn(cfg, "s1", "sqlite foreign keys default", "Off, per the docs.")) == 1
        _wait(lambda: bool(_events(cfg, "knowledge_dispute")))
        _wait(d.settled)
        it = d.knowledge.item(ids.item_id)
        assert it.disputes == 1 and it.valid
        (line,) = _turn(cfg, "s2", "sqlite foreign keys default", "Off again.")
        assert "[disputed " in line and "the docs say they are off by default" in line
        _wait(lambda: len(_events(cfg, "knowledge_dispute")) == 2)
        _wait(d.settled)
        it = d.knowledge.item(ids.item_id)
        assert not it.valid and "contradicted 2 times" in it.invalid_reason
        assert _turn(cfg, "s3", "sqlite foreign keys default", "Off.") == []
    first, second = _events(cfg, "knowledge_dispute")
    assert (first["disputes"], first["retracted"]) == (1, False)
    assert (second["disputes"], second["retracted"]) == (2, True)
    h = memory_help(read_events(events_path(cfg.home)), tz=UTC)["judged"]
    assert h["contradicted"] == 2 and h["misleading_rate"] == 1.0
    assert verdicts('{"A1": "Contradicted"}', [ids]) == {ids.item_id: ("contradicted", "")}
    found = verdicts('{"A1": {"label": "useful", "wrong": "true", "why": "  docs  say "}}', [ids])
    assert found == {ids.item_id: ("contradicted", "docs say")}
    assert verdicts('{"A1": {"label": "useful", "wrong": false}}', [ids]) == {
        ids.item_id: ("useful", "")
    }


def test_what_nobody_checked_is_always_judged() -> None:
    from graph_memory.knowledge import Origin, Volatility, at_risk

    told = Item(type=ItemType.LEARNED, statement="x", origin=Origin.ASSISTANT)
    read = Item(type=ItemType.LEARNED, statement="x", origin=Origin.DOCS)
    versioned = Item(type=ItemType.FACT, statement="x", volatility=Volatility.VERSIONED)
    decided = Item(type=ItemType.DECISION, statement="x", origin=Origin.ASSISTANT)
    assert at_risk(told) and at_risk(versioned)
    assert not at_risk(read) and not at_risk(decided)


def test_stable_notes_are_judged_only_when_sampled(cfg: Config) -> None:
    lock = Item(type=ItemType.DECISION, statement="the limiter uses a redis lua script lock")
    calls: list[str] = []
    with judged_daemon(cfg, lambda p: calls.append(p) or "{}", [lock], sample=1.0) as d:
        assert len(_turn(cfg, "s", "redis lua lock", "lua")) == 1
        _wait(d.settled)
    assert calls == [] and _events(cfg, "knowledge_judge") == []
