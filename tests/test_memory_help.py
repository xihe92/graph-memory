"""How often memory helps: the numbers from the event log (`gm stats`, the Memory help
page) and the daemon bookkeeping behind them."""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.client import DaemonUnavailable, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon
from graph_memory.events import EventLog, events_path
from graph_memory.knowledge import Entity, Item, ItemType, KnowledgeStore
from graph_memory.stats import format_help, memory_help
from graph_memory.webui import UIServer

from .conftest import topic_embed, turn
from .test_daemon import _start, _wait
from .test_graduation import Script
from .test_monitoring import _req


def _recall(ts: str, session: str, items: list[str], **kw: Any) -> dict[str, Any]:
    return {"ts": ts, "kind": "knowledge_recall", "session": session, "mode": "on",
            "items": items, **kw}  # fmt: skip


def _use(ts: str, session: str, recalled: list[str], used: list[str], **kw: Any) -> dict[str, Any]:
    return {"ts": ts, "kind": "knowledge_use", "session": session, "mode": "on",
            "recalled": recalled, "used": used, **kw}  # fmt: skip


EVENTS = [
    _recall("2026-11-04T10:00:00+00:00", "s1", ["a", "b"], prompt="fix the limiter"),
    {"ts": "2026-11-04T10:00:01+00:00", "kind": "placement", "session": "s1"},
    _use("2026-11-04T10:01:00+00:00", "s1", ["a", "b"], ["a"], node="n1"),
    _recall(
        "2026-11-04T11:00:00+00:00", "s1", [], reason="nothing in the prompt points into the graph"
    ),
    # s2's first turn is interrupted: its reply never comes, so it stays unchecked
    _recall("2026-11-05T09:00:00+00:00", "s2", ["b"]),
    _recall("2026-11-05T10:00:00+00:00", "s2", ["b"]),
    _use("2026-11-05T10:01:00+00:00", "s2", ["b"], []),
    # shadow mode: computed, not shown, so not counted as help
    {**_recall("2026-11-05T11:00:00+00:00", "s3", ["a"]), "mode": "shadow"},
    {**_use("2026-11-05T11:01:00+00:00", "s3", ["a"], ["a"]), "mode": "shadow"},
    # a use check that doesn't match what was shown is not credited to the turn
    _recall("2026-11-05T12:00:00+00:00", "s1", ["a"]),
    _use("2026-11-05T12:01:00+00:00", "s1", ["zzz"], ["zzz"]),
    _recall("2026-11-05T13:00:00+00:00", "s4", [], reason="ok"),
]


def test_memory_help_counts_notes_shown_and_used() -> None:
    h = memory_help(EVENTS, tz=UTC)
    t = h["totals"]
    assert (t["prompts"], t["with_notes"], t["notes"]) == (6, 4, 5)
    assert (t["checked"], t["helped"], t["notes_checked"], t["used"]) == (2, 1, 3, 1)
    assert t["unchecked"] == 2
    assert (t["notes_rate"], t["help_rate"], t["use_rate"]) == (round(4 / 6, 3), 0.5, 0.333)
    assert [(d["day"], d["prompts"], d["helped"]) for d in h["days"]] == [
        ("2026-11-04", 2, 1),
        ("2026-11-05", 4, 0),
    ]
    assert h["most_used"] == [{"item": "a", "used": 1, "shown": 1}]
    assert h["never_used"] == [{"item": "b", "shown": 2}]
    assert h["nothing_shown"] == {
        "nothing in the prompt points into the graph": 1,
        "best matches already shown": 1,
    }
    assert h["shadow"] == 1
    newest, *_, oldest = h["turns"]
    assert newest["session"] == "s4" and newest["shown"] == []
    assert oldest["prompt"] == "fix the limiter" and oldest["used"] == ["a"]
    assert oldest["node"] == "n1"
    assert len(memory_help(EVENTS, tz=UTC, max_turns=2)["turns"]) == 2


def test_days_are_in_the_local_time_zone() -> None:
    pacific = timezone(timedelta(hours=-8))
    h = memory_help([_recall("2026-11-05T03:00:00+00:00", "s", [])], tz=pacific)
    assert h["days"][0]["day"] == "2026-11-04"


def test_format_help() -> None:
    items = {"a": {"type": "fact", "statement": "the limiter uses a redis lua script"}}
    text = format_help(memory_help(EVENTS, tz=UTC), items)
    assert "prompts        6   got notes 4 (67%), 5 notes" in text
    assert "checked 2: used a note 1 (50%); notes used 1 of 3 (33%)" in text
    assert "not checked    2" in text and "by day" in text
    assert "1× of 1  [fact] the limiter uses a redis lua script  (a)" in text
    assert "shown, never used\n    2×  b" in text
    assert "shadow mode    1 prompts" in text
    assert "no prompts with knowledge recall on" in format_help(memory_help([]))
    shadow_only = format_help(memory_help(EVENTS[7:9]))
    assert "1 computed in shadow mode" in shadow_only


# --------------------------------------------------------------------------- the daemon


def test_a_restarted_daemon_still_checks_the_reply(short_dir: Path) -> None:
    """The notes waiting for their reply come back from the event log after a restart;
    abandoned and finished turns, and recalls older than six hours, do not."""
    now = datetime.now(UTC)

    def ts(minutes: float) -> str:
        return (now - timedelta(minutes=minutes)).isoformat()

    lines = [
        _recall(ts(60 * 7), "old", ["a"]),  # too old to still be waiting
        _recall(ts(30), "done", ["a"]),
        _use(ts(29), "done", ["a"], []),
        _recall(ts(20), "dropped", ["a"]),
        _recall(ts(19), "dropped", []),  # the next prompt had no notes
        _recall(ts(5), "waiting", ["a", "b"]),
    ]
    events_path(short_dir).write_text("".join(json.dumps(e) + "\n" for e in lines))
    cfg = load_config(short_dir)
    logged = EventLog(events_path(short_dir))
    assert Daemon(cfg, events=logged)._recalled == {"waiting": ["a", "b"]}
    assert Daemon(cfg, events=EventLog(None))._recalled == {}  # no log


@pytest.fixture
def recall_on(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Daemon]]:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    k = KnowledgeStore(short_dir, topic_embed)
    redis = k.add_entity(Entity(name="Redis", kind="tool"))
    k.add_item(
        Item(
            type=ItemType.FACT,
            statement="the limiter uses a redis lua script for acquire",
            item_id="aaaa1111",
            entities=[redis.entity_id],
        )
    )
    k.save()
    cfg = replace(load_config(), save_interval=0.05, knowledge_recall="on")
    daemon, t = _start(cfg, Script([]))
    try:
        yield cfg, daemon
    finally:
        with contextlib.suppress(DaemonUnavailable, OSError):
            call(cfg, "shutdown", autostart=False)
        t.join(5)


def _events(cfg: Config, kind: str) -> list[dict[str, Any]]:
    lines = events_path(cfg.home).read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["kind"] == kind]


def test_the_use_check_follows_the_prompt_it_belongs_to(recall_on: tuple[Config, Daemon]) -> None:
    cfg, daemon = recall_on
    prompt = "why does the redis lua script in acquire  fail?"
    res = call(cfg, "recall", {"session": "s", "prompt": prompt}, autostart=False)
    assert "redis lua script" in json.dumps(res)
    (ev,) = _events(cfg, "knowledge_recall")
    assert ev["prompt"] == "why does the redis lua script in acquire fail?" and ev["items"]
    assert daemon._recalled == {"s": ["aaaa1111"]}

    # interrupted: the next prompt gets no notes, so nothing is left to check against
    # the reply that follows it
    call(cfg, "recall", {"session": "s", "prompt": "weather in seoul"}, autostart=False)
    assert daemon._recalled == {}

    call(cfg, "recall", {"session": "t", "prompt": prompt}, autostart=False)
    m = call(
        cfg,
        "remember",
        {"session": "t", "turn": turn(prompt, "acquire's redis lua script now retries")},
        autostart=False,
    )
    _wait(lambda: bool(_events(cfg, "knowledge_use")))
    (use,) = _events(cfg, "knowledge_use")
    assert use["node"] == m["exchange"] and use["used"] == ["aaaa1111"]


# --------------------------------------------------------------------------- gm stats, web


def test_gm_stats_and_the_help_page(
    recall_on: tuple[Config, Daemon], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, _ = recall_on
    now = datetime.now(UTC)
    lines = [
        _recall(now.isoformat(), "s", ["aaaa1111"], prompt="redis lua"),
        _use(now.isoformat(), "s", ["aaaa1111"], ["aaaa1111"]),
    ]
    with events_path(cfg.home).open("a") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in lines)

    assert cli.main(["stats", "--since", "all"]) == 0
    out = capsys.readouterr().out
    assert "MEMORY HELP" in out
    assert "1× of 1  [fact] the limiter uses a redis lua script for acquire" in out
    assert cli.main(["stats", "--since", "all", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["memory_help"]["totals"]["helped"] == 1

    ui = UIServer(cfg, "127.0.0.1", 0)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    try:
        status, page = _req(ui, "/help")
        assert status == 200 and ui.token.encode() in page and b"Memory <span>Help" in page
        assert _req(ui, "/api/help")[0] == 403
        status, data = _req(ui, "/api/help?since=all", token=ui.token)
        assert status == 200 and data["mode"] == "on"
        assert data["totals"]["helped"] == 1
        assert data["items"]["aaaa1111"]["type"] == "fact"
        assert data["turns"][0]["prompt"] == "redis lua"
        assert _req(ui, "/api/help?since=yesterday", token=ui.token)[0] == 400
    finally:
        ui.shutdown()
        ui.server_close()


def test_odd_timestamps_and_no_daemon(short_dir: Path) -> None:
    assert memory_help([_recall("not a time", "s", [])])["days"][0]["day"] == "not a time"
    assert cli.knowledge_items(load_config(short_dir)) == {}  # no daemon: ids only
